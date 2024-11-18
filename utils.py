import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import scipy
import scipy.io

def plot_noise_distribution(noise, predicted_noise,epoch,plot_path=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4)) 

    # Plot Ground Truth Noise
    axes[0].plot(noise.cpu().numpy()[0, 0], label='GT Noise L', linewidth=0.5, marker='o', markersize=1)
    axes[0].plot(noise.cpu().numpy()[0, 1], label='GT Noise R', linewidth=0.5, marker='o', markersize=1)
    axes[0].grid()
    axes[0].legend()

    # Plot Predicted Noise
    axes[1].plot(predicted_noise.cpu().numpy()[0, 0], label='Pred Noise L', linewidth=0.5, marker='o', markersize=1)
    axes[1].plot(predicted_noise.cpu().numpy()[0, 1], label='Pred Noise R', linewidth=0.5, marker='o', markersize=1)
    axes[1].grid()
    axes[1].legend()

    # Plot Noise Distribution
    axes[2].hist(noise.cpu().numpy().flatten(), density=True, alpha=0.8, label="Ground Truth Noise")
    axes[2].hist(predicted_noise.cpu().numpy().flatten(), density=True, alpha=0.8, label="Predicted Noise")
    axes[2].legend()

    fig.suptitle(f'Noise distribution epoch: {epoch}')
    plt.show()

    if plot_path:
        plt.savefig(plot_path)
        plt.close()



def plot_hrir(hrir_test, hrir_pred, position, id, plot_path):

    plt.figure(figsize=(14, 5))
    plt.plot(hrir_test[0], label='Sample', linestyle='dashed')
    plt.plot(hrir_pred[0], label='Predicted')
    plt.grid(True)
    plt.legend()
    plt.title(f'Left HRIR position: {position}')
    plt.xlabel('Sample Idex')
    plt.ylabel('Amplitude')
    plt.savefig(plot_path)
    plt.close()
    
def hrir2hrtf(hrir_test, hrir_pred, id, plot_path):

    hrtf_l_test = np.fft.fft(hrir_test[:,0])
    hrtf_l_pred = np.fft.fft(hrir_pred[:,0])

    critical_bands = np.array(
        [200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700,
         3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])

    hrtf_l_test_db = 20 * np.log10(np.abs(hrtf_l_test))
    hrtf_l_pred_db = 20 * np.log10(np.abs(hrtf_l_pred))

    K = np.fft.fftfreq(len(hrtf_l_test[0]), 1 / 44100)

    nearest_indices = np.array([np.abs(K - freq).argmin() for freq in critical_bands])
    hrtf_l_test_db_1 = hrtf_l_test_db[:,nearest_indices]
    hrtf_l_pred_db_1 = hrtf_l_pred_db[:,nearest_indices]

    positions = np.random.randint(36, size=10).astype(int)

    directory = f''
    os.makedirs(directory, exist_ok=True)

    for position in positions:

        plt.figure(figsize=(14, 5))
        plt.plot(K[:len(K) // 2], hrtf_l_test_db[position, :len(hrtf_l_test_db[position]) // 2], label='Test')
        plt.plot(K[:len(K) // 2], hrtf_l_pred_db[position, :len(hrtf_l_pred_db[position]) // 2], label='Predicted')
        plt.title(f'Left HRTF source position: {position}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(20, 20000)
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(directory + f'/sub_{id}/hrtf_l_pos_{position}_all.jpg')
        plt.close()

        plt.figure(figsize=(14, 5))
        plt.plot(hrir_pred[position, 0], label='Predicted')
        plt.plot(hrir_test[position,0], label='Sample', linestyle='dashed')
        plt.grid(True)
        plt.legend()
        plt.title(f'Left HRIR position: {position}')
        plt.xlabel('Sample Idex')
        plt.ylabel('Amplitude')
        plt.savefig(directory + f'/sub_{id}/hrir_l_pos_{position}.jpg')
        plt.close()

    return hrtf_l_pred_db_1, hrtf_l_test_db_1


def error_freq(hrir_pred, hrir_test):

    ratio = (hrir_pred.float() - hrir_test.float()) ** 2

    return ratio


def energy_loss(predicted, target):
    return torch.sum((torch.sum(predicted ** 2) - torch.sum(target ** 2)) ** 2)

