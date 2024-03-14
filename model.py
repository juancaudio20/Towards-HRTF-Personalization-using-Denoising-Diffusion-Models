import torch
from torch import nn
import math


class DiffusionModel:
    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=300):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps


        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def forward(self, x_0, t, device="cpu"):

        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)

        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):

        labels = kwargs.get('labels', None)
        head_embedding = kwargs.get('head_embedding', None)
        ears_embedding = kwargs.get('ears_embedding', None)
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        denoise_model = model(x, t, labels=labels, head_embedding=head_embedding, ears_embedding=ears_embedding)
        # print("model out:", denoise_model)
        mean = sqrt_recip_alphas_t * (x - betas_t * denoise_model / sqrt_one_minus_alphas_cumprod_t)
        # mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

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



class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, head_embedding, ears_embedding,
                 num_filters=3, downsample=True):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        self.head_embedding = head_embedding
        self.ears_embedding = ears_embedding
        # self.head_measurement_embedding = nn.Linear(13, 128)

        if labels:
            self.label_mlp = nn.Linear(1, channels_out)

        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv1d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv1d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv1d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose1d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm1d(channels_out)
        self.bnorm2 = nn.BatchNorm1d(channels_out)

        self.conv2 = nn.Conv1d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)

        if ears_embedding:
            self.ears_measurement_embedding = nn.Linear(24, time_embedding_dims)
            self.ears_mlp = nn.Linear(time_embedding_dims, channels_out)

        if head_embedding:
            self.head_measurement_embedding = nn.Linear(13, time_embedding_dims)
            self.head_mlp = nn.Linear(time_embedding_dims, channels_out)
            # self.head_mlp = nn.Linear(13,time_embedding_dims)

        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time.unsqueeze(2)
        if self.head_embedding:
            head_meas = kwargs.get('head_embedding')
            # head_emb = self.head_measurement_embedding(head_meas)
            o_head = self.head_mlp(self.head_measurement_embedding(head_meas.float()))
            # o_head = self.head_mlp(head_meas.float())
            # print("o head:", o_head.shape)
            # print("o: ", o.shape)
            o = o + o_head.unsqueeze(2)
        if self.ears_embedding:
            ear_meas = kwargs.get('ears_embedding')
            o_ears = self.ears_mlp(self.ears_measurement_embedding(ear_meas.float()))
            # print("o_ears", o_ears.shape)
            o = o + o_ears.unsqueeze(2)
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label.unsqueeze(2)
            # print(o)

        return self.final(o)


class UNet(nn.Module):
    def __init__(self, audio_channels=2, time_embedding_dims=256, labels=False, head_embedding=False,
                 ears_embedding=False, sequence_channels=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        sequence_channels_rev = reversed(sequence_channels)

        self.downsampling = nn.ModuleList(
            [Block(channels_in, channels_out, time_embedding_dims, labels, head_embedding, ears_embedding) for
             channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels, head_embedding,
                                               ears_embedding, downsample=False) for channels_in, channels_out in
                                         zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv1d(audio_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv1d(sequence_channels[0], audio_channels, 1)

    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)