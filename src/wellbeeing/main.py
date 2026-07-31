import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import sounddevice as sd
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
from pathlib import Path
import os

# Set Plotting Theme
sns.set_theme(style="white", palette=None)

# Define Color Palette and Cycle for Visualizations
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(color_pal)


def get_data(file):
    """
    Load raw PCM audio data.
    Returns normalized floating-point signal.
    """
    
    signal = np.fromfile(file, dtype=np.int16)
    y = signal.astype(np.float32)

    # Normalize amplitude [-1, 1] (Max absolute value normalization)
    y /= np.max(np.abs(y))

    return y

def play(audio):
    # Plays audio sample
    print("start playing audio sample")
    sd.play(audio, 16000)
    sd.wait()
    print("End of audio sample")

def plot_raw_waveform(y):
    pd.Series(y).plot(
        figsize=(12, 7),             
        lw=1,                       
        title="Raw Audio Waveform (Signal)", 
        color=color_pal[0],         
        xlabel="Time (Samples)",      
        ylabel="Amplitude"          
    )
    plt.show()

def plot_trim_waveform(y):
    """
    Trim the audio signal to remove silence
    `y_trim`: Audio data after silence trimming
    `top_db`: Threshold in decibels for considering a region as silence
    """
    y_trim, _ = librosa.effects.trim(y, top_db=35)

    # Plot the trimmed audio waveform
    pd.Series(y_trim).plot(
        figsize=(12, 7),                
        lw=1,                          
        title="Trimmed Audio Waveform (Silence Removed)",  
        color=color_pal[1],            
        xlabel="Sample Index (Trimmed)", 
        ylabel="Amplitude"             
    )
    plt.show()


def mel_spectrogram(y, sr):

    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=1024
    )

    mel_spect_db = librosa.power_to_db(
        mel_spect,
        ref=np.max
    )

    librosa.display.specshow(
        mel_spect_db,
        y_axis='mel',
        x_axis='time',
        sr=sr
    )

    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_mel(y,sr):

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    print(f"Mel Spectrogram Shape: {S.shape}")

    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    print(f"Decibel Range: {S_db_mel.min():.2f} dB to {S_db_mel.max():.2f} dB")

    # Plot the Log-Scaled Mel Spectrogram
    fig, ax = plt.subplots(figsize=(12, 7))  
    img = librosa.display.specshow(
        S_db_mel,               # Spectrogram data (in dB scale)
        sr=sr,                  # Sampling rate for accurate time display
        x_axis='time',          # X-axis: Time (seconds)
        y_axis='mel',           # Y-axis: Mel-frequency scale
        cmap='magma',           # Use a perceptually uniform colormap
        ax=ax                   # Plot on the specified subplot
    )

    # Add Title and Axis Labels
    ax.set_title("Mel Spectrogram (Log Scale in Decibels)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Mel Frequency (Hz)", fontsize=14)

    # Add a Color Bar to Show the Decibel Range
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Amplitude (dB)", fontsize=12)

    # Adjust Layout for Better Visualization
    plt.tight_layout()
    plt.show()

def spectral_flux(y):
    """
    Calculation of the Euclidean distance between the two normalised spectra
    """

    X = np.abs(librosa.stft(y=y))
    # Normalise each frame
    X /= np.sum(X, axis=0, keepdims=True) + 1e-10
    diff = np.diff(X, axis=1)

    flux = np.linalg.norm(diff, axis=0)

    return flux

def audio_feature_extract(y, sr, recording_id):
    """ 
    Feature extraction for audio recordings. Function returns mean and std value of every 
    feture.

    Inputs: 
    y - signal
    sr - sample rate

    Returns:
    Audio fetures dictionary
    """

    features = {"recording_id":recording_id, "sample rate":sr}

    ### Time domain ###
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = zcr.mean()
    features["zcr_std"] = zcr.std()

    # RMS Energy
    rms = librosa.feature.rms(y=y)

    features["rms_mean"] = rms.mean()
    features["rms_std"] = rms.std()

    ### Frequency domain ###
    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc_{i+1}_std"] = mfcc[i].std()

    # Spectral Centroid
    spect_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    features["spect_centroid_mean"] = spect_centroid.mean()
    features["spect_centroid_std"] = spect_centroid.std()

    # Spectral bandwidth
    spect_bdw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    features["spect_bandwidth_mean"] = spect_bdw.mean()
    features["spect_bandwidth_std"] = spect_bdw.std()

    # Spectral Flux
    flux = spectral_flux(y=y)

    features["flux_mean"] = flux.mean()
    features["flux_std"] = flux.std()
    
    return features


if __name__ == "__main__":
    # Audio processing and extracting features
    y = get_data("data/raw/sound/2_2025_08_08-15_28_39")
    sr = 16000

    f = audio_feature_extract(y,sr, "2_2025_08_08-15_28_39")
    print("Features ", f)

    mel_spectrogram(y, sr)
    plot_mel(y,sr)

    # Generelized approach for all samples
    audio_folder = Path("data/raw/sound")
    
    all_features = []

    for file in audio_folder.iterdir():
        y = get_data(file)
        features = audio_feature_extract(y, 16000, file.name)
        all_features.append(features)

    df = pd.DataFrame(all_features)
    print("DATA FRAME:")
    print(df.head())
    print(df['flux_mean'])

    # Save to csv
    df.to_csv("data/processed/audio_features.csv", index=False)

   


    
    
   

   