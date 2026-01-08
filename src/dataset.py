import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def to_mono(x: np.ndarray) -> np.ndarray:
    # soundfile returns shape (N,) for mono or (N, C) for multi-channel
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def normalize_audio(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 0.0
    if m == 0:
        return x
    return (x / m) * peak

# ---- parameters ----
target_snr_db = 30  # Signal-to-Noise Ratio in dB: higher = quieter noise
random_inject = False  # If True: inject signal randomly in noise; If False: signal starts at beginning
min_start_time = 0.4  # Minimum time (in seconds) from start before signal can appear (only for random_inject)

# ---- paths ----
windows_wavs = list(Path("data/sounds").glob("*.wav"))
rir_wavs = list(Path("data/RIRS_NOISES/real_rirs_isotropic_noises").glob("*.wav"))
noise_wavs = list(Path("data/background").glob("*.wav"))

windows_sounds_to_take = [
    'chimes.wav',
    'chord.wav',
    'Windows Logon.wav',
    'Windows Error.wav',
    'Windows Ding.wav'
]
background_noises_to_take = [
    'noise_30.wav',
    'noise_43.wav',
    'noise_87.wav',
    'noise_244.wav',
    'noise_230.wav'
]

room_to_samples = defaultdict(int)

os.makedirs("output", exist_ok=True)

# total = len(windows_wavs) * len(rir_wavs) * len(noise_wavs)
total = len(rir_wavs) * len(windows_sounds_to_take) * len(background_noises_to_take)

prog = tqdm(total=total)

for rir_wav_path in rir_wavs:
    for windows_wav_path in windows_wavs:
        if windows_wav_path.name not in windows_sounds_to_take:
            continue  # skip unwanted window sounds
        for noise_wav_path in noise_wavs:
            if noise_wav_path.name not in background_noises_to_take:
                continue  # skip unwanted background noises
            out_wav_path = f"{windows_wav_path.stem}_in_room_{rir_wav_path.stem}_with_{noise_wav_path.stem}.wav"
            # ---- load ----
            x, fs_x = sf.read(windows_wav_path)
            h, fs_h = sf.read(rir_wav_path)

            x = to_mono(np.asarray(x, dtype=np.float64))
            h = to_mono(np.asarray(h, dtype=np.float64))

            # ---- ensure same sample rate ----
            if fs_x != fs_h:
                # resample RIR to match audio sample rate (or vice versa)
                # resample_poly is good quality and avoids some FFT resampling artifacts
                gcd = np.gcd(fs_h, fs_x)
                up = fs_x // gcd
                down = fs_h // gcd
                h = resample_poly(h, up, down)
                fs_h = fs_x

            # ---- (optional) trim leading silence in RIR so time-0 is the direct path ----
            # A common trick: find first sample above a small threshold.
            thr = 1e-4 * np.max(np.abs(h)) if np.max(np.abs(h)) > 0 else 0
            if thr > 0:
                idx0 = np.argmax(np.abs(h) > thr)
                h = h[idx0:]

            # ---- convolve (this is the paper’s core step: x_r[t] = x[t] * h_s[t]) ----
            y = fftconvolve(x, h, mode="full")
            # ---- add background noise ----
            # Load noise file
            noise, fs_noise = sf.read(noise_wav_path)
            noise = to_mono(np.asarray(noise, dtype=np.float64))

            # Resample noise if needed
            if fs_noise != fs_x:
                gcd = np.gcd(fs_noise, fs_x)
                up = fs_x // gcd
                down = fs_noise // gcd
                noise = resample_poly(noise, up, down)

            # Match noise length to signal length
            if len(noise) < len(y):
                # Repeat/tile noise if it's shorter than the signal
                num_repeats = int(np.ceil(len(y) / len(noise)))
                noise = np.tile(noise, num_repeats)[:len(y)]
                
                # Calculate RMS and scale noise
                rms_signal = np.sqrt(np.mean(y**2))
                rms_noise = np.sqrt(np.mean(noise**2))
                snr_linear = 10 ** (target_snr_db / 20)
                noise_scale = rms_signal / (rms_noise * snr_linear) if rms_noise > 0 else 0
                
                # Add scaled noise to signal
                y = y + noise * noise_scale
            else:
                # Signal is shorter than noise - place signal in noise
                # Calculate RMS for scaling
                rms_signal = np.sqrt(np.mean(y**2))
                rms_noise = np.sqrt(np.mean(noise**2))
                snr_linear = 10 ** (target_snr_db / 20)
                noise_scale = rms_signal / (rms_noise * snr_linear) if rms_noise > 0 else 0
                
                # Choose position based on parameter
                if random_inject:
                    # Place signal at random position in noise, respecting minimum start time
                    min_start_sample = int(min_start_time * fs_x)
                    max_start = len(noise) - len(y)
                    
                    # Ensure min_start_sample doesn't exceed max_start
                    min_start_sample = min(min_start_sample, max_start)
                    
                    if max_start > min_start_sample:
                        random_start = np.random.randint(min_start_sample, max_start + 1)
                    else:
                        random_start = min_start_sample if min_start_sample >= 0 else 0
                else:
                    # Place signal at min_start_time
                    min_start_sample = int(min_start_time * fs_x)
                    max_start = len(noise) - len(y)
                    random_start = min(min_start_sample, max_start) if max_start >= 0 else 0
                    random_start = max(0, random_start)  # Ensure non-negative
                
                # Create output by placing signal in noise at chosen position
                y_full = noise * noise_scale
                y_full[random_start:random_start + len(y)] += y
                y = y_full
            # ---- normalize to avoid clipping ----
            y = normalize_audio(y, peak=0.99)

            # ---- write result ----
            sf.write("output/" + out_wav_path, y.astype(np.float32), fs_x)

            prog.update(1)