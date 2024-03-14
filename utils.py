import matplotlib.pyplot as plt
import numpy


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