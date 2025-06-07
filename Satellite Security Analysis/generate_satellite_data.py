import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Function to generate synthetic multivariate time-series data
def generate_time_series(n_series, length, base_means, base_stds, label, 
                         normal_means=None, normal_stds=None, 
                         compromised_means=None, compromised_stds=None):
    features = list(base_means.keys())
    data = []
    epsilon = 1e-6  # Small constant to avoid division by zero

    for i in range(n_series):
        series_data = {}
        for feat in features:
            base = base_means[feat]
            std = base_stds[feat]
            scale = max(std / 3, epsilon)  # Standard deviation of noise

            # Label 2: compromised (combination of issue and normal)
            if label == 2:
                # Occasionally resemble normal behavior with a slight drift
                if np.random.rand() < 0.25 and normal_means is not None and normal_stds is not None:
                    base = normal_means[feat]
                    std = normal_stds[feat]
                    scale = max(std / 3, epsilon)
                    values = base + np.cumsum(np.random.normal(0, scale * 0.15, size=length))
                    drift_magnitude = max(abs(std), epsilon) * np.random.uniform(0, 0.15)
                    values += np.linspace(0, drift_magnitude, length)
                    lower_bound = base - 2 * std
                    upper_bound = base + 2 * std + drift_magnitude
                    values = np.clip(values, lower_bound, upper_bound)
                else:
                    # Generate time-series with various compromised behavior patterns
                    noise = np.random.normal(0, scale, size=length)
                    values = base + np.cumsum(noise)
                    mix_type = np.random.choice(['normal_like', 'issue_like', 'extreme'], p=[0.4, 0.4, 0.2])

                    if mix_type == 'normal_like':
                        drift_magnitude = max(abs(std), epsilon) * np.random.uniform(0.15, 0.5)
                        values += np.linspace(0, drift_magnitude, length)
                        values += np.random.normal(0, scale * 0.4, size=length)
                    elif mix_type == 'issue_like':
                        drift_magnitude = max(abs(std), epsilon) * np.random.uniform(0.5, 1.2)
                        values += np.linspace(0, drift_magnitude, length)
                        values += np.random.normal(0, scale * 1.0, size=length)
                    else:  # extreme
                        drift_magnitude = max(abs(std), epsilon) * np.random.uniform(1.0, 2.0)
                        values += np.linspace(0, drift_magnitude, length)
                        spike_times = np.random.choice(length, size=2, replace=False)
                        spike_mean = drift_magnitude * 0.3
                        spike_std = max(drift_magnitude / 8, epsilon)
                        values[spike_times] += np.random.normal(spike_mean, spike_std, size=2)

                    lower_bound = base - 2 * std
                    upper_bound = base + 3 * std + drift_magnitude
                    values = np.clip(values, lower_bound, upper_bound)

            # Label 1: system issue (stronger noise and spikes)
            elif label == 1:
                noise = np.random.normal(0, scale, size=length)
                values = base + np.cumsum(noise)
                extra_noise = np.random.normal(0, scale * 2.0, size=length)
                values += extra_noise
                spike_times = np.random.choice(length, size=2, replace=False)
                spike_magnitude = scale * 4
                values[spike_times] += np.random.normal(spike_magnitude, spike_magnitude / 2, size=2)

            # Label 0: normal behavior (clean, minor noise, small rare spikes)
            else:
                base_jitter = np.random.normal(0, std * 0.4)  # Simulate slight sensor calibration shifts
                base = base + base_jitter
                noise = np.random.normal(0, scale, size=length)
                values = base + np.cumsum(noise)
                if np.random.rand() < 0.4:
                    spike_times = np.random.choice(length, size=1, replace=False)
                    spike_magnitude = scale * 3
                    values[spike_times] += np.random.normal(spike_magnitude, spike_magnitude / 4, size=1)

            series_data[feat] = values

        # Convert series dictionary to DataFrame with metadata
        df_series = pd.DataFrame(series_data)
        df_series['class_label'] = label
        df_series['series_id'] = i
        df_series['time_step'] = np.arange(length)
        data.append(df_series)

    return pd.concat(data, ignore_index=True)

# Define means and standard deviations for each scenario
normal_means = {
    'voltage_main': 30,
    'battery_temp': 30,
    'solar_panel_output': 70,
    'cpu_load': 45,
    'gyro_drift': 0.1,
    'comm_signal_strength': -80,
    'latency_ms': 130
}
normal_stds = {k: abs(v)*0.2 for k, v in normal_means.items()}

system_issue_means = {
    'voltage_main': 28,
    'battery_temp': 40,
    'solar_panel_output': 50,
    'cpu_load': 65,
    'gyro_drift': 0.4,
    'comm_signal_strength': -85,
    'latency_ms': 230
}
system_issue_stds = {k: abs(v)*0.3 for k, v in system_issue_means.items()}

compromised_means = {
    'voltage_main': 29,
    'battery_temp': 35,
    'solar_panel_output': 60,
    'cpu_load': 55,
    'gyro_drift': 0.25,
    'comm_signal_strength': -82,
    'latency_ms': 180
}
compromised_stds = {k: abs(v)*0.25 for k, v in compromised_means.items()}

# Length of each time series
length = 100

# Generate datasets for each class
normal_df = generate_time_series(80, length, normal_means, normal_stds, 0)
issue_df = generate_time_series(30, length, system_issue_means, system_issue_stds, 1)
compromised_df = generate_time_series(30, length, compromised_means, compromised_stds, 2,
                                     normal_means=normal_means, normal_stds=normal_stds,
                                     compromised_means=compromised_means, compromised_stds=compromised_stds)

# Combine and shuffle all data
df = pd.concat([normal_df, issue_df, compromised_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("synthetic_satellite_data.csv", index=False)
print("Synthetic data saved to synthetic_satellite_data.csv")
