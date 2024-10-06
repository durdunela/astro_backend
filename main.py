import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
from scipy.signal import butter, filtfilt
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import sounddevice as sd
from scipy.signal import resample

def load_catalog(cat_file):
    """Load the catalog from a CSV file."""
    return pd.read_csv(cat_file)

def parse_event_row(row):
    """Parse an event row from the catalog."""
    # Arrival time (When the moonquake happened)
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
    # Relative time (How long after we started recording the moonquake we detected it)
    arrival_time_rel = row['time_rel(sec)']
    test_filename = row['filename']  # The name of a file that contains the moonquake data
    return arrival_time, arrival_time_rel, test_filename

def load_seismic_data(data_directory, filename):
    """Load seismic data from a CSV file."""
    csv_file = f'{data_directory}{filename}.csv'
    data = pd.read_csv(csv_file)
    csv_times = np.array(data['time_rel(sec)'].tolist())
    csv_data = np.array(data['velocity(m/s)'].tolist())
    return csv_times, csv_data

def plot_seismic_trace(csv_times, csv_data, arrival_time_rel, filename):
    """Plot the seismic trace."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(csv_times, csv_data)
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{filename}', fontweight='bold')

    arrival_line = ax.axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
    ax.legend(handles=[arrival_line])
    plt.show()

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def isolate_noise(csv_times, filtered_data, noise_start_time, noise_end_time):
    """
    Isolate the noise part of the filtered data based on start and end times.
    
    Parameters:
        csv_times (numpy array): The array of time values.
        filtered_data (numpy array): The filtered seismic data.
        noise_start_time (float): The start time of the noise part.
        noise_end_time (float): The end time of the noise part.

    Returns:
        noise_times (numpy array): The time values corresponding to the noise part.
        noise_data (numpy array): The noise data.
    """
    # Find indices for the noise time range
    start_idx = np.where(csv_times >= noise_start_time)[0][0]
    end_idx = np.where(csv_times <= noise_end_time)[0][-1]
    
    # Slice the data to get the noise part
    noise_times = csv_times[start_idx:end_idx + 1]
    noise_data = filtered_data[start_idx:end_idx + 1]
    
    return noise_times, noise_data


def play_seismic_data(csv_times, seismic_data, playback_speed=100, fs_audio=44100):
    """
    Play the seismic data as sound.

    Parameters:
        csv_times (numpy array): The array of time values (seconds).
        seismic_data (numpy array): The array of seismic velocity data.
        playback_speed (float): How much to speed up the seismic data to fit within an audible time frame.
        fs_audio (int): The audio sampling rate (default is 44.1 kHz).
    """
    # Resample the seismic data to fit the audio sample rate (fs_audio)
    num_samples = int(len(seismic_data) * fs_audio / (csv_times[-1] - csv_times[0]) / playback_speed)
    seismic_data_resampled = resample(seismic_data, num_samples)

    # Normalize the seismic data to fit within -1 to 1 for audio playback
    seismic_data_resampled = seismic_data_resampled / np.max(np.abs(seismic_data_resampled))

    # Play the sound
    print("Playing seismic data as sound...")
    sd.play(seismic_data_resampled, fs_audio)
    sd.wait()  # Wait until the sound finishes playing

def process_event(cat_file, data_directory, row_idx):
    """Process a single event from the catalog."""
    cat = load_catalog(cat_file)
    row = cat.iloc[row_idx]

    arrival_time, arrival_time_rel, test_filename = parse_event_row(row)

    csv_times, csv_data = load_seismic_data(data_directory, test_filename)

    plot_seismic_trace(csv_times, csv_data, arrival_time_rel, test_filename)

    # Set parameters for filtering
    fs = 1.0 / (csv_times[1] - csv_times[0])  # Sampling frequency
    lowcut = 0.5
    highcut = 1.0

    # Filter the seismic data
    filtered_data = bandpass_filter(csv_data, lowcut, highcut, fs)

    # Isolate noise part
    noise_start_time = 10  # Adjust this to your specific noise start time
    noise_end_time = 30    # Adjust this to your specific noise end time
    noise_times, noise_data = isolate_noise(csv_times, filtered_data, noise_start_time, noise_end_time)

    # Create a spectrogram to see the patterns in the filtered data
    f, t, sxx = signal.spectrogram(filtered_data, fs)

    # Plot the time series and spectrogram
    fig = plt.figure(figsize=(10, 10))

    # Plot 1: Time Series of the Filtered Data
    ax = plt.subplot(2, 1, 1)
    ax.plot(csv_times, filtered_data, label='Filtered Data')
    ax.axvline(x=arrival_time_rel, color='red', label='Detection')
    ax.legend(loc='upper left')
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    # Plot 2: Spectrogram
    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(csv_times), max(csv_times)])
    ax2.set_xlabel('Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.axvline(x=arrival_time_rel, c='red')

    # Color bar for the spectrogram
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
    
    plt.show()

    # Plot the isolated noise part
    plt.figure(figsize=(10, 3))
    plt.plot(noise_times, noise_data, color='black')
    plt.xlim([min(noise_times), max(noise_times)])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Isolated Noise Part', fontweight='bold')
    plt.grid()
    plt.show()

    # Play the seismic data as sound
    play_seismic_data(csv_times, filtered_data, playback_speed=100)

    # STA/LTA parameters
    sta_len = 120  # Short-term window length in seconds
    lta_len = 600  # Long-term window length in seconds

    # Compute the STA/LTA characteristic function
    cft = classic_sta_lta(filtered_data, int(sta_len * fs), int(lta_len * fs))

    # Plot characteristic function
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(csv_times, cft)
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')
    plt.title('STA/LTA Characteristic Function')
    plt.grid()

    # Play around with the on and off triggers, based on values in the characteristic function
    thr_on = 4  # Trigger on threshold
    thr_off = 1.5  # Trigger off threshold
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))

    # Plot on and off triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for i in np.arange(0, len(on_off)):
        triggers = on_off[i]
        ax.axvline(x=csv_times[triggers[0]], color='red', label='Trig. On' if i == 0 else "")
        ax.axvline(x=csv_times[triggers[1]], color='purple', label='Trig. Off' if i == 0 else "")

    # Plot the filtered seismic data
    ax.plot(csv_times, filtered_data, label='Filtered Seismic Data', color='black')
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    plt.title('Seismic Data with Trigger Markers')
    plt.grid()

    plt.show()

# Directories and file names
cat_directory = '/home/shurgi/Downloads/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
data_directory = '/home/shurgi/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'

# Process the event at the specified index
process_event(cat_file, data_directory, 6)