import pandas as pd
import numpy as np

# Sample Kp index dataset (time_tag, kp, observed, noaa_scale)
kp_raw = [
    ["time_tag", "kp", "observed", "noaa_scale"],
    ["2025-05-20 00:00:00", "2.33", "observed", None],
    ["2025-05-20 03:00:00", "3.00", "observed", None],
    ["2025-05-20 06:00:00", "3.00", "observed", None],
    ["2025-05-20 09:00:00", "3.00", "observed", None],
    ["2025-05-20 12:00:00", "3.00", "observed", None],
    ["2025-05-20 15:00:00", "2.67", "observed", None],
    ["2025-05-20 18:00:00", "2.00", "observed", None],
    ["2025-05-20 21:00:00", "3.00", "observed", None],
    ["2025-05-21 00:00:00", "4.00", "observed", None],
    ["2025-05-21 03:00:00", "3.00", "observed", None],
    ["2025-05-21 06:00:00", "3.00", "observed", None],
    ["2025-05-21 09:00:00", "3.00", "observed", None],
    ["2025-05-21 12:00:00", "2.33", "observed", None],
    ["2025-05-21 15:00:00", "1.33", "observed", None],
    ["2025-05-21 18:00:00", "1.67", "observed", None],
    ["2025-05-21 21:00:00", "2.33", "observed", None],
    ["2025-05-22 00:00:00", "2.00", "observed", None],
    ["2025-05-22 03:00:00", "2.00", "observed", None],
    ["2025-05-22 06:00:00", "2.00", "observed", None],
    ["2025-05-22 09:00:00", "1.33", "observed", None],
    ["2025-05-22 12:00:00", "1.67", "observed", None],
    ["2025-05-22 15:00:00", "1.33", "observed", None],
    ["2025-05-22 18:00:00", "2.33", "observed", None],
    ["2025-05-22 21:00:00", "1.67", "observed", None],
    ["2025-05-23 00:00:00", "2.00", "observed", None],
    ["2025-05-23 03:00:00", "1.67", "observed", None],
    ["2025-05-23 06:00:00", "2.00", "observed", None],
    ["2025-05-23 09:00:00", "2.00", "observed", None],
    ["2025-05-23 12:00:00", "2.33", "observed", None],
    ["2025-05-23 15:00:00", "2.33", "observed", None],
    ["2025-05-23 18:00:00", "1.33", "observed", None],
    ["2025-05-23 21:00:00", "2.00", "observed", None],
    ["2025-05-24 00:00:00", "1.67", "observed", None],
    ["2025-05-24 03:00:00", "1.33", "observed", None],
    ["2025-05-24 06:00:00", "1.67", "observed", None],
    ["2025-05-24 09:00:00", "1.33", "observed", None],
    ["2025-05-24 12:00:00", "1.67", "observed", None],
    ["2025-05-24 15:00:00", "1.00", "observed", None],
    ["2025-05-24 18:00:00", "0.67", "observed", None],
    ["2025-05-24 21:00:00", "1.33", "observed", None],
    ["2025-05-25 00:00:00", "1.67", "observed", None],
    ["2025-05-25 03:00:00", "2.00", "observed", None],
    ["2025-05-25 06:00:00", "2.00", "observed", None],
    ["2025-05-25 09:00:00", "1.33", "observed", None],
    ["2025-05-25 12:00:00", "2.00", "observed", None],
    ["2025-05-25 15:00:00", "1.33", "observed", None],
    ["2025-05-25 18:00:00", "1.00", "observed", None],
    ["2025-05-25 21:00:00", "1.00", "observed", None],
    ["2025-05-26 00:00:00", "1.67", "observed", None],
    ["2025-05-26 03:00:00", "1.67", "observed", None],
    ["2025-05-26 06:00:00", "2.00", "observed", None],
    ["2025-05-26 09:00:00", "1.67", "observed", None],
    ["2025-05-26 12:00:00", "3.00", "observed", None],
    ["2025-05-26 15:00:00", "1.33", "observed", None],
    ["2025-05-26 18:00:00", "2.00", "observed", None],
    ["2025-05-26 21:00:00", "2.67", "observed", None],
    ["2025-05-27 00:00:00", "2.33", "observed", None],
    ["2025-05-27 03:00:00", "2.33", "observed", None],
    ["2025-05-27 06:00:00", "1.67", "observed", None],
    ["2025-05-27 09:00:00", "2.67", "observed", None],
    ["2025-05-27 12:00:00", "3.00", "observed", None],
    ["2025-05-27 15:00:00", "2.67", "observed", None],
]

# Convert raw Kp data into DataFrame
kp_df = pd.DataFrame(kp_raw[1:], columns=kp_raw[0])
kp_df['time_tag'] = pd.to_datetime(kp_df['time_tag'])
kp_df['kp'] = kp_df['kp'].astype(float)

# Load satellite data (ensure the file has an 'epoch' column)
sat_df = pd.read_excel('tle_with_parsed_features.xlsx')  # Update with correct file name if needed
sat_df['epoch'] = pd.to_datetime(sat_df['epoch'])

# Match closest Kp index to each satellite epoch
kp_times = kp_df['time_tag'].values  # numpy datetime64 array

def find_closest_kp_index(ts):
    ts_np = np.datetime64(ts)
    return np.abs(kp_times - ts_np).argmin()

closest_kp_indices = [find_closest_kp_index(ts) for ts in sat_df['epoch']]
sat_df['kp_index'] = [kp_df.iloc[i]['kp'] for i in closest_kp_indices]

# Save results to Excel
output_filename = 'satellite_data_with_kp_index.xlsx'
sat_df.to_excel(output_filename, index=False)

print(f"Saved updated satellite data with matched kp_index to {output_filename}")
