#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Signal Processing Script
Input: CSV file from inference.py (containing keypoint trajectories)
Process: Differentiation -> Filtering -> FT -> STFT -> MSE
Output: Processed signals and extracted features (Peak Frequency, CIA, etc.)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import pyentrp.entropy as ent
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path, keypoint_of_interest='right_knee', direction='y'):
    """
    Load the keypoints CSV and extract the trajectory of a specific keypoint and direction.
    Args:
        csv_path (str): Path to the CSV file from inference.py.
        keypoint_of_interest (str): Name of the keypoint (e.g., 'right_knee').
        direction (str): 'x' or 'y'.
    Returns:
        t (np.array): Time vector (assuming 30 FPS).
        y_original (np.array): The original displacement signal.
    """
    df = pd.read_csv(csv_path)
    # Construct column name (e.g., 'right_knee_y')
    col_name = f"{keypoint_of_interest}_{direction}"
    # Extract the signal, convert to numpy array, handle potential NaNs
    y_original = df[col_name].values.astype(float)
    # Simple linear interpolation to fill NaNs (if any)
    nans = np.isnan(y_original)
    if nans.any():
        y_original[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], y_original[~nans])
    
    # Create time vector (assuming 30 Hz sampling rate)
    Fs = 30.0  # Sampling Frequency (Hz)
    t = np.arange(len(y_original)) / Fs # Time in seconds

    return t, y_original

def differentiate_and_filter(t, y_original, cutoff_freq=3.0, fs=30.0):
    """
    Calculate acceleration from displacement and apply a low-pass filter.
    Args:
        t (np.array): Time vector.
        y_original (np.array): Displacement signal.
        cutoff_freq (float): Cutoff frequency for the low-pass filter (Hz).
        fs (float): Sampling frequency (Hz).
    Returns:
        y_accel (np.array): Filtered acceleration signal.
    """
    # First derivative -> velocity, Second derivative -> acceleration
    # Use central differences for numerical differentiation
    velocity = np.gradient(y_original, t)
    acceleration = np.gradient(velocity, t)

    # Design a low-pass FIR filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    numtaps = 101  # Filter order (length)
    b = signal.firwin(numtaps, normal_cutoff, window='hamming')
    
    # Apply the filter to the acceleration signal
    y_accel = signal.filtfilt(b, 1.0, acceleration) # filtfilt for zero-phase filtering

    return y_accel

def compute_fft(t, y_signal, fs=30.0):
    """
    Perform Fourier Transform to find the dominant frequency.
    Args:
        t (np.array): Time vector.
        y_signal (np.array): Input signal (e.g., acceleration).
        fs (float): Sampling frequency.
    Returns:
        freqs (np.array): Frequency bins.
        fft_magnitude (np.array): Magnitude of the FFT.
        peak_freq (float): The frequency with the highest magnitude.
    """
    N = len(t)
    yf = fft(y_signal)
    xf = fftfreq(N, 1 / fs)[:N//2] # Positive frequencies only

    fft_magnitude = 2.0/N * np.abs(yf[0:N//2])
    peak_index = np.argmax(fft_magnitude)
    peak_freq = xf[peak_index]

    return xf, fft_magnitude, peak_freq

def compute_stft(t, y_signal, fs=30.0, nperseg=256):
    """
    Perform Short-Time Fourier Transform to visualize time-frequency changes.
    Args:
        t (np.array): Time vector.
        y_signal (np.array): Input signal.
        fs (float): Sampling frequency.
        nperseg (int): Length of each segment for STFT.
    Returns:
        f_stft (np.array): Array of sample frequencies.
        t_stft (np.array): Array of segment times.
        Zxx (np.array): STFT of `y_signal`.
    """
    f_stft, t_stft, Zxx = signal.stft(y_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)
    return f_stft, t_stft, Zxx

def find_change_point_from_stft(t_stft, Zxx, freq_band_of_interest=(0.5, 3.0)):
    """
    Heuristic method to find a significant change point from the STFT spectrogram.
    This is a simplified example. You might need a more sophisticated method.
    Args:
        t_stft (np.array): Time bins from STFT.
        Zxx (np.array): STFT matrix.
        freq_band_of_interest (tuple): (low_freq, high_freq) band to monitor.
    Returns:
        change_point_index (int): Index in the `t_stft` array where the change occurs.
    """
    # Find indices of frequencies within the band of interest
    f_stft = # ... (needs to be passed in or calculated from STFT parameters)
    idx_band = np.where((f_stft >= freq_band_of_interest[0]) & (f_stft <= freq_band_of_interest[1]))[0]
    
    # Calculate total energy in the band over time
    energy_in_band = np.sum(np.abs(Zxx[idx_band, :]), axis=0)
    
    # Smooth the energy signal
    energy_smooth = np.convolve(energy_in_band, np.ones(5)/5, mode='same')
    
    # Find the point of maximum negative gradient (steepest drop)
    gradient = np.gradient(energy_smooth)
    change_point_index = np.argmin(gradient) # Index of the steepest drop
    
    # Convert STFT time index back to approximate original time index
    change_time = t_stft[change_point_index]
    return change_point_index, change_time

def calculate_multiscale_entropy(y_signal, max_scale=20):
    """
    Calculate Multiscale Entropy (MSE) for a signal.
    Args:
        y_signal (np.array): Input signal.
        max_scale (int): Maximum scale factor.
    Returns:
        scales (list): List of scale factors.
        entropy_values (list): Sample Entropy for each scale.
    """
    scales = range(1, max_scale+1)
    entropy_values = []
    for s in scales:
        # Coarse-graining procedure
        coarse_grained = []
        for i in range(0, len(y_signal) // s):
            coarse_grained.append(np.mean(y_signal[i*s : (i+1)*s]))
        coarse_grained = np.array(coarse_grained)
        
        # Calculate Sample Entropy for the coarse-grained series
        # Using common parameters: embedding dimension m=2, tolerance r=0.2 * std
        if len(coarse_grained) > 10: # Ensure enough data points
            samp_en = ent.sample_entropy(coarse_grained, 2, 0.2 * np.std(coarse_grained))
            # sample_entropy returns a list for m and m+1, we take the value for m=2
            entropy_values.append(samp_en[0])
        else:
            entropy_values.append(np.nan)
    
    return scales, entropy_values

def calculate_cia(entropy_values, scales):
    """
    Calculate Complexity Index Average (CIA).
    CIA = (1/max_scale) * sum(SE for scale 1 to max_scale)
    Args:
        entropy_values (list): Sample Entropy values for each scale.
        scales (list): List of scale factors.
    Returns:
        cia (float): The Complexity Index Average.
    """
    # Use only valid (non-NaN) entropy values
    valid_entropy = [en for en in entropy_values if not np.isnan(en)]
    if not valid_entropy:
        return np.nan
    cia = np.mean(valid_entropy)
    return cia

if __name__ == "__main__":
    # --- Configuration ---
    CSV_PATH = "path/to/your/keypoints_data.csv"
    OUTPUT_DIR = "path/to/your/processed/output"

    # --- Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    t, disp_y = load_and_preprocess_data(CSV_PATH, keypoint_of_interest='right_knee', direction='y')

    # --- Differentiate and Filter to get Acceleration ---
    print("Calculating acceleration...")
    accel = differentiate_and_filter(t, disp_y, cutoff_freq=3.0, fs=30.0)

    # --- Save Processed Acceleration Data ---
    processed_df = pd.DataFrame({'Time (s)': t, 'Acceleration (pixels/s^2)': accel})
    processed_df.to_csv(os.path.join(OUTPUT_DIR, 'processed_acceleration.csv'), index=False)
    print(f"Processed acceleration data saved.")

    # --- Fourier Transform to find dominant pedaling frequency ---
    print("Performing Fourier Transform...")
    freqs, fft_mag, pf = compute_fft(t, accel, fs=30.0)
    print(f"Peak Frequency: {pf:.2f} Hz")

    # --- STFT to find change point ---
    print("Performing STFT...")
    f_stft, t_stft, Zxx = compute_stft(t, accel, fs=30.0, nperseg=256) # nperseg ~8.5 seconds at 30Hz
    change_point_index, change_time = find_change_point_from_stft(t_stft, Zxx, (0.5, 3.0))
    print(f"Estimated change point at time: {change_time:.2f} seconds")

    # --- Multiscale Entropy Analysis ---
    print("Calculating Multiscale Entropy...")
    scales, entropy_vals = calculate_multiscale_entropy(accel, max_scale=20)
    cia = calculate_cia(entropy_vals, scales)
    print(f"Complexity Index Average (CIA): {cia:.3f}")

    # --- (Optional) Plotting ---
    # ... You can add code here to generate plots similar to your paper's figures ...

    print("Signal processing complete.")