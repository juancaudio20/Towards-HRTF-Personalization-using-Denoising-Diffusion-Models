import torch
from torch import nn
import math
from tqdm import tqdm
import torch.nn.functional as F



class DiffusionModel:
    def __init__(self, noise_steps=600, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.device = device
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def forward(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def backward(self, x, model, **kwargs):
        labels = kwargs.get('labels', None)
        head_embedding = kwargs.get('head_embedding', None)
        ears_embedding = kwargs.get('ears_embedding', None)
        cfg_scale = 1
        model.eval()
        with torch.inference_mode():
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(len(labels)) * i).long().to(self.device)
                predicted_noise = model(x, t, labels=labels, head_embedding=head_embedding, ears_embedding=ears_embedding)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, labels=None, head_embedding=None, ears_embedding=None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x
    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        out = values.gather(-1, t.long().cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)





class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels, dropout_prob=0.2):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True, dropout=dropout_prob)
        self.ln = nn.LayerNorm(channels)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(channels, channels),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = self.attn_dropout(attention_value)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        attention_value = attention_value.permute(0, 2, 1)

        return attention_value



class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, head_embedding, ears_embedding,
                 kernel_size=3, downsample=True, dropout_prob=0.0):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        self.head_embedding = head_embedding
        self.ears_embedding = ears_embedding

        if labels:
            self.label_emb = nn.Embedding(labels, channels_out)

        self.downsample = downsample

        padding = kernel_size // 2

        if downsample:
            self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size, padding=padding)
            self.final = nn.Conv1d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv1d(2 * channels_in, channels_out, kernel_size, padding=padding)
            self.final = nn.ConvTranspose1d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm1d(channels_out)
        self.bnorm2 = nn.BatchNorm1d(channels_out)

        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)

        if ears_embedding:
            self.ears_measurement_embedding = nn.Linear(12, channels_out)

        if head_embedding:
            self.head_measurement_embedding = nn.Linear(13, channels_out)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time.unsqueeze(2)
        head_meas = kwargs.get('head_embedding')
        ear_meas = kwargs.get('ears_embedding')
        label = kwargs.get('labels')
        if head_meas is not None:
            o_head = self.relu(self.head_measurement_embedding(head_meas))
            o = o + o_head.unsqueeze(2)
        if ear_meas is not None:
            o_ears = self.relu(self.ears_measurement_embedding(ear_meas))
            o = o + o_ears.unsqueeze(2)
        if label is not None:
            o_label = self.relu(self.label_emb(label))
            o_label = o_label.squeeze(1)
            o = o + o_label.unsqueeze(2)

        o = self.dropout(self.bnorm2(self.relu(self.conv2(o))))

        return self.final(o)


class UNet(nn.Module):
    def __init__(self, audio_channels=1, time_embedding_dims=128, labels=False, ears_embedding=False, head_embedding=False, sequence_channels=(4, 8, 16, 32, 64, 128), dropout_prob=0.0):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims

        self.downsampling = nn.ModuleList(
            [Block(channels_in, channels_out, time_embedding_dims, labels, head_embedding, ears_embedding, dropout_prob=dropout_prob) for
             channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels, head_embedding, ears_embedding, downsample=False, dropout_prob=dropout_prob) for channels_in, channels_out in
                                         zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv1d(audio_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv1d(sequence_channels[0], audio_channels, 1)

        self.attentions = nn.ModuleList([SelfAttention(channels_out, dropout_prob) for channels_out in sequence_channels[:-1]])

    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)

        for us, res, attention in zip(self.upsampling, reversed(residuals), reversed(self.attentions)):
            if o.shape[2] != res.shape[2]:
                if o.shape[2] < res.shape[2]:
                    padding = res.shape[2] - o.shape[2]
                    o = F.pad(o, (0, padding))
                else:
                    padding = o.shape[2] - res.shape[2]
                    res = attention(F.pad(res, (0, padding)))

            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)



class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


