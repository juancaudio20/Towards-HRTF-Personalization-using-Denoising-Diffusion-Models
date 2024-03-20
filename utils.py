import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_noise_distribution(noise, predicted_noise,epoch,plot_path=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))  # Create three subplots

    # Plot GT Noise
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

def nmse(hrir_test, hrir_gen):

    sq_error_left = torch.mean((hrir_test[0] - hrir_gen[0]) ** 2)
    sq_error_right = torch.mean((hrir_test[1] - hrir_gen[1]) ** 2)

    power_left = torch.mean(hrir_test[0] ** 2)
    power_right = torch.mean(hrir_test[1] ** 2)

    nmse_left = sq_error_left / power_left
    nmse_right = sq_error_right / power_right

    overall_nmse = (nmse_left + nmse_right) / 2

    return overall_nmse


def lsd(hrir_test, hrir_gen, points,sr, plot=False):

    hrtf_array1 = []
    hrtf_array2 = []

    for point in range(points):
        hrtf_test = np.fft.fft(hrir_test[point][0])
        hrtf_gen = np.fft.fft(hrir_gen[point][0])
        K = np.fft.fftfreq(len(hrtf_test), 1 / sr)

        hrtf_array1.append(hrtf_gen)
        hrtf_array2.append(hrtf_test)


    H = np.array(hrtf_array1)
    H_hat = np.array(hrtf_array2)

    log_ratio = 20 * np.log10(np.abs(hrtf_array1) / np.abs(hrtf_array2))

    squared_diffs = np.sum(log_ratio ** 2) / (points * len(K))
    LSD = np.sqrt(squared_diffs)
    '''
    cumulative_LSD = np.zeros(K)
    cumulative_LSD[0] = np.sqrt(np.sum(log_ratio[:, :1] ** 2) / (points * 1))

    for i in range(1, K):
        log_ratio_i = log_ratio[:, :i + 1]
        squared_diffs_i = np.sum(log_ratio_i ** 2) / (points * (i + 1))
        cumulative_LSD[i] = np.sqrt(squared_diffs_i)

    frequency_axis = np.arange(1, K + 1) * (44100 / (2 * K))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(frequency_axis, cumulative_LSD, linewidth=0.5)
        plt.title('Cumulative LSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Cumulative LSD [dB]')
        plt.grid(True)
        plt.xscale('log')
        plt.show()
    '''
    return LSD

