import torch
import argparse
import tqdm
import numpy as np

from dataset import HUTUBSDataset, collate_fn
from model import DiffusionModel, UNet
from utils import plot_noise_distribution


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/allpos_1403_curie")


parser = argparse.ArgumentParser(description='Training script with configurable parameters')
parser.add_argument('--BATCH_SIZE', type=int, default=128, help='Data batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--early_stop_patience', type=int, default=200, help='Patience for early stopping')
parser.add_argument('--saved_model_path', type=str, default='/nas/home/jalbarracin/ddpm/saved models/training_0903/train1403_curie', help='Path to save trained model')
parser.add_argument('--plot_path',type=str,default='/nas/home/jalbarracin/ddpm/saved models/training_0903/noise_distribution_curie.jpg')
args = parser.parse_args()


hutubs_dataset = HUTUBSDataset(
    hrtf_directory='/nas/home/jalbarracin/datasets/HUTUBS/HRIRs',
    anthro_csv_path='/nas/home/jalbarracin/datasets/HUTUBS/AntrhopometricMeasures.csv'
)


indices_to_remove = []
for n in range(92):
    for p in range(440):
        hutubs_inspect = hutubs_dataset[n * 440 + p]
        if hutubs_inspect['subject_id'] in [18,79,92]:
            indices_to_remove.append(n * 440 + p)

for idx in sorted(indices_to_remove, reverse=True):
    hutubs_dataset.normalized_dataset.pop(idx)

print("dataset len", len(hutubs_dataset))


train_percentage = 0.7  # 70% for training, 20% for validation, 10% for test
val_percentage = 0.2
test_percentage = 0.1

# Calculate the split index
train_split_index = int(len(hutubs_dataset) * train_percentage)
val_split_index = train_split_index + int(len(hutubs_dataset)*val_percentage)

# Split the dataset into training and testing sets
train_dataset = hutubs_dataset[:train_split_index]
val_dataset = hutubs_dataset[train_split_index:val_split_index]
test_dataset = hutubs_dataset[val_split_index:]

print("Training set: ",len(train_dataset))
print("Validation set: ", len(val_dataset))
print("Test set: ", len(test_dataset))

NO_EPOCHS = args.epochs
LR = args.lr
VERBOSE = args.verbose
early_stop_patience = args.early_stop_patience
saved_model_path = args.saved_model_path
BATCH_SIZE = args.BATCH_SIZE
plot_path = args.plot_path

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True,collate_fn=collate_fn)


diffusion_model = DiffusionModel()

unet = UNet(labels=True,head_embedding=True,ears_embedding=True)
unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)


if VERBOSE:
    print(f'Number of epochs: {NO_EPOCHS}')
    print(f'Learning rate: {LR}')
    print(f'Early stop patience: {early_stop_patience}')
    print(f'Saved model path: {saved_model_path}')

for epoch in tqdm.tqdm(range(NO_EPOCHS),desc='Training', unit='epoch'):
    mean_epoch_loss = []
    mean_epoch_loss_val = []

    for data in train_loader:
        unet.train(True)
        batch = data['hrtf'].to(device)
        label = data['measurement_point'].to(device).float()
        # print(label)
        head_measurements = data['head_measurements'].to(device)
        ears_measurements = data['ear_measurements'].to(device)
        # head_measurements = torch.round(head_measurements*10000).long()
        # print("shape head: ", head_measurements.shape)

        # print("shape HE:", head_encoded.shape)
        # af_encoded = head_enc(af_label)
        # print(head_encoded)
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)

        # print("head: ", head_measurements_offsets.shape)
        '''
        batch = batch.to(device)
        print("batch: ", batch.shape)
        print("meas: ", label.shape)
        print("af: ", af_label.shape)
        print("t: ", t.shape)
        '''
        batch_noisy, noise = diffusion_model.forward(batch, t, device)
        # batch_noisy = batch_noisy.unsqueeze(1).expand(-1, 2, -1)
        batch_noisy = batch_noisy.float()
        # print("batch_noisy:", batch_noisy.shape)
        # print("t",t.shape)
        predicted_noise = unet(batch_noisy, t, labels=label.reshape(-1, 1), head_embedding=head_measurements,
                               ears_embedding=ears_measurements)
        # predicted_noise = unet(batch_noisy, t, labels=label.reshape(-1, 1))
        optimizer.zero_grad()
        train_loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        mean_epoch_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

    for data in val_loader:
        unet.eval()
        batch = data['hrtf'].to(device)
        label = data['measurement_point'].to(device).float()
        head_measurements = data['head_measurements'].to(device)
        ears_measurements = data['ear_measurements'].to(device)
        # head_measurements = torch.round(head_measurements*10000).long()
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).float().to(device)
        # batch = batch.to(device)
        # batch = batch.float()

        batch_noisy, noise = diffusion_model.forward(batch, t, device)
        batch_noisy = batch_noisy.float()
        # print("batch noisy: ",batch_noisy.shape)
        predicted_noise = unet(batch_noisy, t, labels=label.reshape(-1, 1), head_embedding=head_measurements,
                               ears_embedding=ears_measurements)

        val_loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        mean_epoch_loss_val.append(val_loss.item())

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.flush()
    if epoch == 0:
        val_loss_best = val_loss
        early_stop_counter = 0
        saved_model_path = saved_model_path
        torch.save(unet.state_dict(), saved_model_path)
        print(f"Model saved epoch: {epoch}, val loss: {val_loss}")
        print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
        with torch.no_grad():
            plot_noise_distribution(noise, predicted_noise,epoch,plot_path)

    if epoch > 0 and val_loss < val_loss_best:
        saved_model_path = saved_model_path
        torch.save(unet.state_dict(), saved_model_path)
        val_loss_best = val_loss
        print(f"Model saved epoch: {epoch}, val loss: {val_loss}")
        print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
        with torch.no_grad():
            plot_noise_distribution(noise, predicted_noise,epoch,plot_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print("Patience status:" + str(early_stop_counter) + "/" + str(early_stop_patience))
        # Early stopping
    if early_stop_counter > early_stop_patience:
        print(f"Training finished at epoch: {epoch}")
        break
