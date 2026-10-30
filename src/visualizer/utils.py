import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


 
def plot_acceleration_data(acc, time, start_time, truncate = 0, label="Left Hand"):
    outlier_time = [start_time - time[truncate]]
    acc = gaussian_filter1d(acc, sigma=1)
    
    for t_outlier in outlier_time:
        plt.figure(figsize=(8,4))
        plt.plot(time[:acc.shape[0]], acc, label=label)
        plt.axvline(t_outlier, color='r', linestyle='--', label="Outlier")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration magnitude")
        plt.title(f"Acceleration around {t_outlier + time[truncate]:.2f}s")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'logs/tests/acc_data_{label}.png')
        plt.close()
        
def plot_audio_data(audio_mean, time, start_time, truncate = 0):
    outlier_time = [start_time - time[truncate]]
    time = time - time[0]
    shape = min(audio_mean.shape[0], time.shape[0])

    for t_outlier in outlier_time:
        plt.figure(figsize=(8,4))
        plt.plot(time[:shape], audio_mean[:shape], label="Audio RMS")
        plt.axvline(t_outlier, color='r', linestyle='--', label="Outlier")
        plt.xlabel("Time [s]")
        plt.ylabel("Audio RMS Amplitude")
        plt.title(f"Audio RMS around {t_outlier + time[truncate]:.2f}s")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('logs/tests/audio_data.png')
        plt.close()

def plot_clap_detection(data_roko: dict, aria_audio: dict, peaks_lh, peaks_rh, peaks_audio, start_idx_roko, start_idx_aria, peak_max_threshold=150):
    """
    Plot Rokoko hand accelerations and Aria audio signal over time with detected peaks.
    """
    # --- Rokoko data ---
    time_roko = data_roko["Timestamp"] * 1e-3
    acc_lh = np.sqrt(
        data_roko["LeftHand_acceleration_x"]**2 +
        data_roko["LeftHand_acceleration_y"]**2 +
        data_roko["LeftHand_acceleration_z"]**2
    )
    acc_rh = np.sqrt(
        data_roko["RightHand_acceleration_x"]**2 +
        data_roko["RightHand_acceleration_y"]**2 +
        data_roko["RightHand_acceleration_z"]**2
    )
    acc_lh = replace_exceeding_with_prev(acc_lh, peak_max_threshold)
    acc_rh = replace_exceeding_with_prev(acc_rh, peak_max_threshold)
    acc_lh = gaussian_filter1d(acc_lh, sigma=1.0)
    acc_rh = gaussian_filter1d(acc_rh, sigma=1.0)
    
    # --- Aria audio data ---
    time_aria = aria_audio["timestamp"] * 1e-9
    audio_mean = np.sqrt(np.nanmean(np.square(aria_audio["data"].astype(np.float64)), axis=1))

    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    # Plot Rokoko acceleration
    axs[0].scatter(time_roko[peaks_lh], acc_lh[peaks_lh], color="blue", marker="x", label="LH peaks")
    axs[0].scatter(time_roko[peaks_rh], acc_rh[peaks_rh], color="green", marker="x", label="RH peaks")
    if start_idx_roko is not None:
        axs[0].axvline(time_roko[start_idx_roko], color="red", linestyle="--", label="Detected Start (Rokoko)")
    axs[0].set_title("Rokoko Hand Accelerations")
    axs[0].set_ylabel("Acceleration Magnitude [m/s²]")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].set_ylim(0, 150)

    # Plot Aria audio
    #axs[1].plot(time_aria, audio_mean[:-1], label="Audio RMS", color="orange", alpha=0.8)
    axs[1].scatter(time_aria[peaks_audio], audio_mean[peaks_audio], color="red", marker="x", label="Audio peaks")
    if start_idx_aria is not None:
        axs[1].axvline(time_aria[start_idx_aria], color="red", linestyle="--", label="Detected Start (Aria)")
    axs[1].set_title("Aria Audio RMS Signal")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig('logs/tests/time_sync_data.png')
    plt.close(fig)

def plot_time_sync_drift(time_aria, time_roko, time_roko_rh, roko_time_shift):
    plt.figure(figsize=(10,5))

    time_difference = time_aria - time_roko
    time_difference_2 = time_aria - time_roko_rh
    
    outlier_idx = np.where(time_difference_2 <= -0.1)
    print(time_roko[outlier_idx]+roko_time_shift)

    plt.plot(time_roko, time_difference, label="LH Acceleration")
    plt.plot(time_roko_rh, time_difference_2, label="RH Acceleration")

    plt.xlabel("Time of audio detection [s]")
    plt.ylabel("Time difference between sensors [s]")
    plt.title("Time Synchronization Drift Test")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('logs/tests/time_sync_drift_test.png')

    plt.figure(figsize=(10,5))
    plt.hist([time_difference, time_difference_2], 
            bins=50, 
            color=['skyblue','salmon'], 
            edgecolor='black', 
            range=(-0.2,0.2), 
            stacked=True, 
            label=['Sensor 1', 'Sensor 2'])

    plt.title("Time Drift Sensor Sync")
    plt.xlabel("Time [s]")
    plt.ylabel("Amount")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig('logs/tests/time_sync_drift_histo_test.png')

def plot_acc(time, acc_lh, acc_rh, num_frames_clap_period):
    time_window = time[:num_frames_clap_period]    
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_window, acc_lh, label="Left Hand Acceleration", color='blue')
    plt.plot(time_window, acc_rh, label="Right Hand Acceleration", color='red')
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration Magnitude [m/s²]")
    plt.title("Left vs Right Hand Acceleration Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_vel(time, vel_x_left, vel_y_left, vel_z_left, speed_left, footlock_mask_left, footlock_soft_left, vel_x_right, vel_y_right, vel_z_right, speed_right, footlock_mask_right, footlock_soft_right):
    # --- Plot ---
    # --- Figure setup ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Left foot ---
    axs[0].plot(time, vel_x_left, label="vel_x", color="r", alpha=0.7)
    axs[0].plot(time, vel_y_left, label="vel_y", color="g", alpha=0.7)
    axs[0].plot(time, vel_z_left, label="vel_z", color="b", alpha=0.7)
    axs[0].plot(time, speed_left, label="speed", color="k", linewidth=1.5)
    axs[0].plot(time, footlock_soft_left, label="lock", linewidth=1.5)

    axs[0].fill_between(time, 0, np.max(speed_left),
                        where=footlock_mask_left, color="gray", alpha=0.2, label="Foot Locked")

    axs[0].set_title("Left Foot Velocities and Foot Lock Detection")
    axs[0].set_ylabel("Velocity / Speed [m/s]")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    # --- Right foot ---
    axs[1].plot(time, vel_x_right, label="vel_x", color="r", alpha=0.7)
    axs[1].plot(time, vel_y_right, label="vel_y", color="g", alpha=0.7)
    axs[1].plot(time, vel_z_right, label="vel_z", color="b", alpha=0.7)
    axs[1].plot(time, speed_right, label="speed", color="k", linewidth=1.5)
    axs[1].plot(time, footlock_soft_right, label="lock", linewidth=1.5)

    axs[1].fill_between(time, 0, np.max(speed_right),
                        where=footlock_mask_right, color="gray", alpha=0.2, label="Foot Locked")

    axs[1].set_title("Right Foot Velocities and Foot Lock Detection")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity / Speed [m/s]")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    # plt.show()
    plt.savefig('logs/tests/vel.png')
    plt.close(fig)
    
def replace_exceeding_with_prev(arr, threshold):
    arr = arr.copy()
    mask = arr > threshold
    # Create an array of indices
    idx = np.arange(len(arr))
    # For positions where the value is above threshold, set index to previous valid
    idx[mask] = 0  # temporary
    # Replace mask positions with the last valid index
    for i in range(1, len(arr)):
        if mask[i]:
            idx[i] = idx[i+1]
        else:
            idx[i] = i
    return arr[idx]