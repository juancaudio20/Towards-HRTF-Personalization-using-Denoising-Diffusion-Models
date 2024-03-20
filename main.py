import torch
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio

from dataset import HUTUBSDataset, collate_fn
from model import DiffusionModel, UNet
from utils import plot_noise_distribution, nmse, lsd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter("runs/emlabel_lr2")


parser = argparse.ArgumentParser(description='Training script with configurable parameters')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Data batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for optimizer')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--early_stop_patience', type=int, default=200, help='Patience for early stopping')
parser.add_argument('--saved_model_path', type=str, default='/nas/home/jalbarracin/ddpm/saved models/training_0903/labellr2', help='Path to save trained model')
parser.add_argument('--plot_path',type=str,default='/nas/home/jalbarracin/ddpm/saved models/training_0903/noise_distribution_labellr2.jpg')
parser.add_argument('--training',type=bool,default=False, help='True for training, else inference')
args = parser.parse_args()

#gen_plot_path = '/nas/home/jalbarracin/ddpm/saved models/training_0903/gen_hrir.jpg'

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

print("dataset len:", len(hutubs_dataset))


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
training = args.training
NUM_CLASSES = 440

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True,collate_fn=collate_fn)


diffusion_model = DiffusionModel()

unet = UNet(labels=440,head_embedding=True,ears_embedding=True)
unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
#dummy_input = torch.randn(BATCH_SIZE, 2, 256).to(device)
#t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
##dummy_label = torch.tensor(10).to(device)
##dummy_head = torch.randn(BATCH_SIZE, 13).to(device)
##dummy_ears = torch.randn(BATCH_SIZE, 24).to(device)
#writer.add_graph(unet,dummy_input)
#writer.close()

if VERBOSE:
    print(f'Number of epochs: {NO_EPOCHS}')
    print(f'Learning rate: {LR}')
    print(f'Early stop patience: {early_stop_patience}')
    print(f'Saved model path: {saved_model_path}')

if training:
    for epoch in tqdm.tqdm(range(NO_EPOCHS),desc='Training', unit='epoch'):
        mean_epoch_loss = []
        mean_epoch_loss_val = []

        for data in train_loader:
            unet.train(True)
            batch = data['hrtf'].to(device)
            label = data['measurement_point'].to(device)
            #label = torch.nn.functional.one_hot(label, num_classes=440)
            #label = label.float()
            #print(label.shape)
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
            predicted_noise = unet(batch_noisy, t, labels=label, head_embedding=head_measurements,
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
            label = data['measurement_point'].to(device)
            #exlabel = torch.nn.functional.one_hot(label, num_classes=440)
            #label = label.float()
            head_measurements = data['head_measurements'].to(device)
            ears_measurements = data['ear_measurements'].to(device)
            # head_measurements = torch.round(head_measurements*10000).long()
            t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).float().to(device)
            # batch = batch.to(device)
            # batch = batch.float()

            batch_noisy, noise = diffusion_model.forward(batch, t, device)
            batch_noisy = batch_noisy.float()
            # print("batch noisy: ",batch_noisy.shape)
            predicted_noise = unet(batch_noisy, t, labels=label, head_embedding=head_measurements,
                                   ears_embedding=ears_measurements)

            val_loss = torch.nn.functional.l1_loss(noise, predicted_noise)
            mean_epoch_loss_val.append(val_loss.item())

            #writer.add_scalar("Loss/train", train_loss, epoch)
            #writer.add_scalar("Loss/val", val_loss, epoch)

            #writer.flush()
        if epoch == 0:
            val_loss_best = val_loss
            early_stop_counter = 0
            saved_model_path = saved_model_path
            #torch.save(unet.state_dict(), saved_model_path)
            print(f"Model saved epoch: {epoch}, val loss: {val_loss}")
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
            with torch.no_grad():
                plot_noise_distribution(noise, predicted_noise,epoch,plot_path)

        if epoch > 0 and val_loss < val_loss_best:
            saved_model_path = saved_model_path
            #torch.save(unet.state_dict(), saved_model_path)
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

else:

    #subject_idx = 83
    load_model_path = '/nas/home/jalbarracin/ddpm/saved models/training_0903/labellr2'
    unet = UNet(labels=440, head_embedding=True, ears_embedding=True)
    unet.load_state_dict(torch.load((load_model_path)))
    torch.manual_seed(16)


    '''
    batch = next(iter(test_loader))
    audio_test = batch['hrtf'][0].to(device).float()
    idsub = batch['subject_id'][0].to(device)
    head = batch['head_measurements'][0].to(device)
    ears = batch['ear_measurements'][0].to(device)
    error_sub = []
    head = head.repeat(1,1)
    ears = ears.repeat(1, 1)
    '''
    unet.eval().to(device)
    error_data = []
    lsd_data = []
    error_df = pd.DataFrame(index=range(83, 93), columns=range(440))
    for subject_idx in tqdm.tqdm(range(83,93),desc='Generation', unit='Subject'):
        error_sub = []
        lsd_sub = []
        hrir_sub = []
        hrir_tsub = []
        for c in range(440):
            #print(next(iter(test_loader)))
            data_test = hutubs_dataset[subject_idx * 440 + c]
            subject_id = data_test['subject_id']
            print(f"Subject: {subject_id}, position: {c}")
            hrir_test = data_test['hrtf']
            head = data_test['head_measurements'].to(device)
            ears = data_test['ear_measurements'].to(device)
            head = head.repeat(1, 1)
            ears = ears.repeat(1, 1)
            audio_result = torch.randn((1,) + (2, 256)).to(device)


            for i in reversed(range(diffusion_model.timesteps)):
                t = torch.full((1,), i, dtype=torch.long, device=device)
                labels = torch.tensor([c]).to(device)

                audio_result = diffusion_model.backward(x=audio_result, t=t, model=unet,
                                                        labels=labels, head_embedding=head.float(),
                                                        ears_embedding=ears.float())

            if np.isnan(audio_result.detach().cpu().any()):
                print("NaN!")
            else:
                #print("ok!")

                audio_result = audio_result.detach().cpu()
                error = nmse(hrir_test=hrir_test, hrir_gen=audio_result[0])
                print("error: ", error.item())
                plt.plot(audio_result[0,0], label='L', linewidth=0.5)
                plt.plot(audio_result[0,1], label='R', linewidth=0.5)
                plt.grid()
                plt.legend()
                plt.savefig(f'/nas/home/jalbarracin/ddpm/results/img/gen_hrir_pos_{c}.jpg')
                plt.close()
                hrir_save = (audio_result[0] * data_test['global_std']) + data_test['global_mean']
                torchaudio.save(
                    uri=f'/nas/home/jalbarracin/ddpm/generated_hrir/fixed/sub_{subject_idx}/pos_{c}.wav',
                    src=hrir_save, sample_rate=44100)
                hrir_sub.append(audio_result[0])
                hrir_tsub.append(hrir_test)

            #print(f"Overall error Subject {subject_id}: ", np.mean(error_sub))
            error_sub.append(error.item())
        log_dist = lsd(hrir_tsub, hrir_sub, 440,44100)
        lsd_data.append(log_dist)
        error_data.append(error_sub)
        print(f"Overall error: ", np.mean(error_data))
        print(f"lsd: ", np.mean(lsd_data))

    lsd_df = pd.DataFrame(lsd_data, index=range(83, 93), columns=range(1))
    error_df = pd.DataFrame(error_data)
    lsd_df.to_excel('/nas/home/jalbarracin/ddpm/results/lsd_values.xlsx')
    error_df.to_excel('/nas/home/jalbarracin/ddpm/results/error_values.xlsx')



