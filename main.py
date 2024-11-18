import torch
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio

import random
from dataset import HUTUBSDataset, collate_fn
from model import DiffusionModel, UNet, EMA
from utils import plot_noise_distribution,error_freq, plot_hrir, hrir2hrtf, energy_loss
import scipy.io
from datetime import datetime
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter("runs/LOOCV_17_04")


parser = argparse.ArgumentParser(description='Training script with configurable parameters')
parser.add_argument('--BATCH_SIZE', type=int, default=2048, help='Data batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('--early_stop_patience', type=int, default=300, help='Patience for early stopping')
parser.add_argument('--saved_model_path', type=str, default='/nas/home/jalbarracin/ddpm/saved models/training_2503/LOOCV', help='Path to save trained model')
parser.add_argument('--plot_path',type=str,default='/nas/home/jalbarracin/ddpm/saved models/training_2503/noise_distribution_cont_noise4.jpg')
parser.add_argument('--training',type=bool,default=True, help='True for training, else inference')
parser.add_argument('--lr_decay', type=float, default=0.8, help="decay learning rate")
parser.add_argument('--interval', type=int, default=100, help="interval to decay lr")
args = parser.parse_args()



NO_EPOCHS = args.epochs
LR = args.lr
VERBOSE = args.verbose
early_stop_patience = args.early_stop_patience
saved_model_path = args.saved_model_path
BATCH_SIZE = args.BATCH_SIZE
plot_path = args.plot_path
training = args.training


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

diffusion_model = DiffusionModel()

def adjust_alpha(alpha, decay, interval, epoch_num):
    alpha = alpha * (decay ** (epoch_num//interval))
    return alpha

if VERBOSE:
    print(f'Number of epochs: {NO_EPOCHS}')
    print(f'Learning rate: {LR}')
    print(f'Early stop patience: {early_stop_patience}')
    print(f'Saved model path: {saved_model_path}')

if training:

    total_subjects = 93 
    print("Total subjects: ", total_subjects)

    subject_indices = list(range(total_subjects))

    error_sub_l_fq = []
    error_sub_l_fq2 = []
    error_sub_r_fq = []

    start_time = datetime.now()

    for val_subject_idx in tqdm.tqdm(subject_indices, desc='LOOCV round'):
        val_subject_idx = val_subject_idx + 20
        print("val id: ", val_subject_idx)

        if val_subject_idx not in {17, 78, 91}:
            hutubs_dataset = HUTUBSDataset(
                hrtf_directory='/nas/home/jalbarracin/datasets/HUTUBS/HRIRs',
                anthro_csv_path='/nas/home/jalbarracin/datasets/HUTUBS/AntrhopometricMeasures.csv',
                val_sub_idx=val_subject_idx,
                pad_size=10
            )

            writer = SummaryWriter(f"runs/test_03_right_{val_subject_idx}")
            saved_model_path = '/nas/home/jalbarracin/ddpm/saved models/training_1804/LOO00CV_left'
            unet = UNet(labels=441, head_embedding=True, ears_embedding=True)
            unet.to(device)
            optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
            alpha=1
        

            train_dataset, val_dataset = hutubs_dataset[0], hutubs_dataset[1]
            locv = val_dataset[0]
            std = locv['global_std']
            mean = locv['global_mean']
            #training and validation datasets
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
            early_stop_counter = 0
            best_loss = float('inf')
            ema = EMA(0.995)
            ema.register(unet)
            for epoch in range(NO_EPOCHS):
                mean_epoch_loss = []
                mean_epoch_loss_val = []
                for data in train_loader:
                    unet.train(True)

                    batch = data['hrtf'].to(device)

                    batch = batch[:, 0, :].unsqueeze(1)
                    label = data['point'].long().to(device)
                    head_measurements = data['head_measurements'].float().to(device)
                    ears_measurements = data['ear_measurements'].float().to(device)    
        
                    t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
                    batch_noisy, noise = diffusion_model.forward(batch, t)
                    batch_noisy = batch_noisy.float()

                    if np.random.random() < 0.1:
                        label = None
                        head_measurements = None
                        ears_measurements = None

                    predicted_noise = unet(batch_noisy, t, labels=label, head_embedding=head_measurements,ears_embedding=ears_measurements, dropout_prob=0.2) # simple unet

                    energy_loss = (torch.nn.functional.l1_loss(torch.fft.fft(predicted_noise),torch.fft.fft(noise)))*alpha
                    train_loss = torch.nn.functional.l1_loss(predicted_noise, noise) + energy_loss
                    optimizer.zero_grad()
                    train_loss.backward()

                    mean_epoch_loss.append(train_loss.item())
            
                    optimizer.step()
                    ema.update(unet)

                adjust_learning_rate(args, LR, optimizer, epoch)

                unet.eval()
                with torch.inference_mode():
                    for data in val_loader:
                        batch = data['hrtf'].to(device)
                        batch = batch[:, 0, :].unsqueeze(1)
                        label = data['point'].long().to(device)
                        head_measurements = data['head_measurements'].float().to(device)
                        ears_measurements = data['ear_measurements'].float().to(device)
        
                        t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)

                        batch_noisy, noise = diffusion_model.forward(batch, t)
                        batch_noisy = batch_noisy.float()
                        predicted_noise = unet(batch_noisy, t, labels=label, head_embedding=head_measurements, ears_embedding=ears_measurements, dropout_prob=0.2)
                        energy_loss = (torch.nn.functional.l1_loss(torch.fft.fft(predicted_noise),torch.fft.fft(noise)))*alpha
                        val_loss = torch.nn.functional.l1_loss(predicted_noise, noise) + energy_loss
                        mean_epoch_loss_val.append(val_loss.item())



                epoch_train_loss = np.mean(mean_epoch_loss)
                epoch_val_loss = np.mean(mean_epoch_loss_val)
                writer.add_scalar("Loss/train", epoch_train_loss, epoch)
                writer.add_scalar("Loss/val", epoch_val_loss, epoch)

                writer.flush()
                alpha = adjust_alpha(alpha, 0.1, 100, epoch)

                if epoch_val_loss < best_loss:
                    saved_model_path = f'/nas/home/jalbarracin/ddpm/saved models/training_1006/LO00OCV1_{val_subject_idx}_L'
                    torch.save(unet.state_dict(), saved_model_path)
                    best_loss = epoch_val_loss
                    print(f"Model saved epoch: {epoch}, val loss: {epoch_val_loss}")
                    time_elapsed = datetime.now() - start_time
                    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
                    early_stop_counter = 0
                    mat_noise = f'/nas/home/jalbarracin/ddpm/results/mat_left/noise/noise_sub_{val_subject_idx}_epoch_{epoch}.mat'
                    scipy.io.savemat(mat_noise,
                                     {f'noise_{val_subject_idx}': predicted_noise.detach().cpu(), f'noise_{val_subject_idx}_gt': noise.detach().cpu()})
                else:
                    early_stop_counter += 1
                    time_elapsed = datetime.now() - start_time
                    if early_stop_counter % 10 == 0:
                        print('Patience status: ' + str(early_stop_counter) + '/' + str(early_stop_patience))
                        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

                if early_stop_counter > early_stop_patience:
                    print('Training finished at epoch ' + str(epoch))
                    break

            print("Training complete")
            load_model_path = f'/nas/home/jalbarracin/ddpm/saved models/training_1006/LO00OCV1_{val_subject_idx}_L'
            unet = UNet(labels=441, head_embedding=True, ears_embedding=True)
            unet.load_state_dict(torch.load((load_model_path)))
            torch.manual_seed(16)

            unet.eval().to(device)
            with torch.inference_mode():
                for data in val_loader:

                    batch = data['hrtf'].to(device)
                    batch = batch[:, 0, :].unsqueeze(1)
                    label = data['point'].to(device)
                
                    head_measurements = data['head_measurements'].float().to(device)
                    ears_measurements = data['ear_measurements'].float().to(device)
                
                    audio_result = torch.randn((batch.shape[0],) + (1, 276)).to(device)
                    audio_result = diffusion_model.backward(x=audio_result, model=unet, labels=label, head_embedding=head_measurements, ears_embedding=ears_measurements)
                    audio_result_np = audio_result.detach().cpu().numpy()
                    if np.isnan(audio_result_np).any():
                        print("NaN!")
                    else:
                        audio_result = audio_result.detach().cpu()
                        batch = batch.detach().cpu()
                        audio_result_denorm = (audio_result * std) + mean
                        batch_denorm = (batch * std) + mean
                        hrtf_l_pred, hrtf_l_test, sam1= hrir2hrtf(batch_denorm[:,:,10:-10], audio_result_denorm[:,:,10:-10], val_subject_idx)
                        error_l_fq = error_freq(torch.from_numpy(hrtf_l_pred), torch.from_numpy(hrtf_l_test))
                        mat_path1 = f'/nas/home/jalbarracin/ddpm/results/mat_left/sub_{val_subject_idx}_L.mat'
                        scipy.io.savemat(mat_path1, {f'sub_{val_subject_idx}_pred': audio_result, f'sub_{val_subject_idx}_gt': batch, f'sub_{val_subject_idx}_pred_denorm': audio_result_denorm, f'sub_{val_subject_idx}_gt_denorm': batch_denorm})

                error_sub_l_fq.append(error_l_fq)

            error_data_l_fq = torch.stack(error_sub_l_fq)

            print("LSD: ", torch.sqrt(torch.mean(error_data_l_fq)).item())

            mat_path2 = f'/nas/home/jalbarracin/ddpm/results/mat_left/error/error{val_subject_idx}_L.mat'
            scipy.io.savemat(mat_path2,
                             {f'LSD': torch.sqrt(torch.mean(error_data_l_fq)).item(),
                              f'sub_{val_subject_idx}_error': error_sub_l_fq})
