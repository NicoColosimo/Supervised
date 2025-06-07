import numpy as np
import librosa
import pandas as pd

sr = 22050
base_duration = 5.0  # base duration in seconds

submarine_profiles = {
    "diesel-electric": {
        "base_freq_range": [50, 200],
        "harmonics_count": 4,
        "harmonics_richness": "medium",
        "modulation_freq": [0.5, 2.0],
        "modulation_depth": [0.1, 0.3],
        "cavitation_rate": [5, 15],
        "cavitation_amplitude": [0.2, 0.5],
        "hydrodynamic_band": [50, 500],
        "hydrodynamic_level": "medium",
        "background_noise_level": "medium",
        "context": "snorkeling and coastal transit"
    },
    "nuclear": {
        "base_freq_range": [20, 100],
        "harmonics_count": 3,
        "harmonics_richness": "low",
        "modulation_freq": [0.1, 0.5],
        "modulation_depth": [0.05, 0.2],
        "cavitation_rate": [10, 20],
        "cavitation_amplitude": [0.5, 0.8],
        "hydrodynamic_band": [20, 400],
        "hydrodynamic_level": "high",
        "background_noise_level": "low",
        "context": "open-ocean cruising"
    },
    "fast": {
        "base_freq_range": [100, 1000],
        "harmonics_count": 5,
        "harmonics_richness": "high",
        "modulation_freq": [2.0, 5.0],
        "modulation_depth": [0.2, 0.5],
        "cavitation_rate": [15, 30],
        "cavitation_amplitude": [0.6, 1.0],
        "hydrodynamic_band": [100, 2000],
        "hydrodynamic_level": "very high",
        "background_noise_level": "medium",
        "context": "high-speed transit or attack"
    },
    "ballistic": {
        "base_freq_range": [20, 80],
        "harmonics_count": 2,
        "harmonics_richness": "low",
        "modulation_freq": [0.1, 0.3],
        "modulation_depth": [0.05, 0.2],
        "cavitation_rate": [5, 10],
        "cavitation_amplitude": [0.3, 0.6],
        "hydrodynamic_band": [10, 300],
        "hydrodynamic_level": "low",
        "background_noise_level": "low",
        "context": "deterrent patrol"
    },
    "mini": {
        "base_freq_range": [200, 1000],
        "harmonics_count": 3,
        "harmonics_richness": "high",
        "modulation_freq": [1.0, 5.0],
        "modulation_depth": [0.2, 0.6],
        "cavitation_rate": [10, 20],
        "cavitation_amplitude": [0.4, 0.7],
        "hydrodynamic_band": [100, 2000],
        "hydrodynamic_level": "medium",
        "background_noise_level": "high",
        "context": "coastal/inland operations"
    },
    "deep-diver": {
        "base_freq_range": [30, 150],
        "harmonics_count": 3,
        "harmonics_richness": "medium",
        "modulation_freq": [0.1, 0.5],
        "modulation_depth": [0.1, 0.3],
        "cavitation_rate": [2, 5],
        "cavitation_amplitude": [0.1, 0.4],
        "hydrodynamic_band": [20, 200],
        "hydrodynamic_level": "low",
        "background_noise_level": "low",
        "context": "extreme-depth operations"
    },
    "stealth": {
        "base_freq_range": [20, 50],
        "harmonics_count": 1,
        "harmonics_richness": "very low",
        "modulation_freq": [0.05, 0.2],
        "modulation_depth": [0.0, 0.1],
        "cavitation_rate": [0, 2],
        "cavitation_amplitude": [0.0, 0.3],
        "hydrodynamic_band": [10, 100],
        "hydrodynamic_level": "very low",
        "background_noise_level": "low",
        "context": "silent running mode"
    },
    "experimental": {
        "base_freq_range": [20, 2000],
        "harmonics_count": 10,
        "harmonics_richness": "complex",
        "modulation_freq": [0.1, 10.0],
        "modulation_depth": [0.1, 0.9],
        "cavitation_rate": [5, 25],
        "cavitation_amplitude": [0.5, 1.0],
        "hydrodynamic_band": [10, 5000],
        "hydrodynamic_level": "high",
        "background_noise_level": "variable",
        "context": "experimental propulsion tests"
    },
    "cargo": {
        "base_freq_range": [30, 150],
        "harmonics_count": 2,
        "harmonics_richness": "low",
        "modulation_freq": [0.1, 0.5],
        "modulation_depth": [0.1, 0.3],
        "cavitation_rate": [5, 10],
        "cavitation_amplitude": [0.3, 0.6],
        "hydrodynamic_band": [20, 300],
        "hydrodynamic_level": "medium",
        "background_noise_level": "high",
        "context": "logistics transit"
    },
    "hunter-killer": {
        "base_freq_range": [50, 200],
        "harmonics_count": 4,
        "harmonics_richness": "high",
        "modulation_freq": [1.0, 3.0],
        "modulation_depth": [0.2, 0.6],
        "cavitation_rate": [10, 30],
        "cavitation_amplitude": [0.7, 1.0],
        "hydrodynamic_band": [50, 1000],
        "hydrodynamic_level": "very high",
        "background_noise_level": "medium",
        "context": "ASW/attack profile"
    }
}

# --- Utility to Convert Qualitative Noise Levels to Amplitude Values ---
def level_to_amplitude(level):
    base_map = {
        "very low": 0.01,
        "low": 0.03,
        "medium": 0.07,
        "high": 0.12,
        "very high": 0.2,
        "variable": 0.05,
        "complex": 0.05
    }
    base_amp = base_map.get(level, 0.05)
    jitter = np.random.uniform(0.8, 1.2)  # introduce ±20% random variation
    return base_amp * jitter

# --- Generate the Harmonic and Modulated Signal for the Submarine ---
def generate_profile_signal(profile, sr, duration):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Extract parameters from the profile
    base_f_low, base_f_high = profile["base_freq_range"]
    harm_count = profile["harmonics_count"]
    mod_freq_low, mod_freq_high = profile["modulation_freq"]
    mod_depth_low, mod_depth_high = profile["modulation_depth"]

    # Random harmonic base frequencies
    base_freqs = np.sort(np.random.uniform(base_f_low, base_f_high, harm_count))
    
    # Random amplitude modulation settings
    mod_freq = np.random.uniform(mod_freq_low, mod_freq_high)
    mod_depth = np.random.uniform(mod_depth_low, mod_depth_high)

    signal = np.zeros_like(t)
    for f in base_freqs:
        # Apply sinusoidal amplitude modulation to each harmonic
        mod_signal = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        signal += mod_signal * np.sin(2 * np.pi * f * t)

    signal /= np.max(np.abs(signal))  # Normalize signal
    return signal

# --- Add Cavitation Bursts (Short, Impulse-Like Noises) ---
def add_cavitation_noise(signal, rate_range, amp_range, sr, duration):
    rate = np.random.uniform(rate_range[0], rate_range[1])  # number of pops/sec
    amplitude = np.random.uniform(amp_range[0], amp_range[1])
    num_pops = int(rate * duration)
    pop_length = int(sr * 0.01)  # each pop lasts 10ms

    cav_signal = np.zeros_like(signal)
    pop_positions = np.random.randint(0, len(signal) - pop_length, num_pops)
    for pos in pop_positions:
        pop_noise = amplitude * np.random.uniform(-1, 1, pop_length)
        cav_signal[pos:pos+pop_length] += pop_noise
    return signal + cav_signal

# --- Add Hydrodynamic Noise in a Specific Frequency Band ---
def add_band_limited_noise(signal, band, amplitude, sr):
    noise = np.random.randn(len(signal))  # generate white noise
    fft_noise = np.fft.rfft(noise)  # frequency domain representation
    freqs = np.fft.rfftfreq(len(noise), 1/sr)
    mask = (freqs >= band[0]) & (freqs <= band[1])  # keep only band frequencies
    fft_noise[~mask] = 0
    band_noise = np.fft.irfft(fft_noise, n=len(noise))  # inverse FFT to time domain
    band_noise /= np.max(np.abs(band_noise))  # normalize
    return signal + amplitude * band_noise

# --- Add Background Noise Based on Qualitative Level ---
def add_background_noise(signal, level):
    amp = level_to_amplitude(level)
    noise = amp * np.random.randn(len(signal))
    return signal + noise

# --- Generate Full Signal with All Noise Types Included ---
def generate_profile_signal_with_noise(profile, sr=sr, base_duration=base_duration):
    duration = base_duration + np.random.uniform(-0.5, 0.5)  # ±0.5s variation
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    base_signal = generate_profile_signal(profile, sr, duration)
    signal = base_signal

    # Add cavitation pops
    signal = add_cavitation_noise(signal,
                                   profile["cavitation_rate"],
                                   profile["cavitation_amplitude"],
                                   sr, duration)

    # Add hydrodynamic band-limited noise
    hydro_amp = level_to_amplitude(profile["hydrodynamic_level"])
    signal = add_band_limited_noise(signal,
                                    profile["hydrodynamic_band"],
                                    hydro_amp, sr)

    # Add broadband background noise
    signal = add_background_noise(signal,
                                  profile["background_noise_level"])

    # Normalize final signal
    signal = signal / np.max(np.abs(signal))
    return signal

# --- Extract Audio Features Using Librosa ---
def extract_audio_features(y, sr):
    features = {}
    features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["rms"] = np.mean(librosa.feature.rms(y=y))
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i in range(5):
        features[f"mfcc_{i+1}"] = np.mean(mfccs[i])
    return features

# --- Main: Generate Feature Dataset for Each Submarine Type ---
rows = []
samples_per_class = 100  # how many audio samples to generate per profile

for name, profile in submarine_profiles.items():
    for _ in range(samples_per_class):
        y = generate_profile_signal_with_noise(profile)
        feats = extract_audio_features(y, sr)
        row = {"profile": name}
        row.update(feats)
        rows.append(row)

# Create DataFrame, shuffle, and export
df = pd.DataFrame(rows)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset
df.to_excel("submarine_acoustic_features_with_noise_100samples.xlsx", index=False)
