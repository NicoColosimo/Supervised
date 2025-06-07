import pandas as pd
from datetime import datetime, timedelta

from datetime import datetime, timedelta

def parse_epoch_from_tle1(tle_line1):
    # Year (cols 19-20) and day of year (cols 21-32)
    year = int(tle_line1[18:20])
    year += 2000 if year < 57 else 1900
    day_of_year = float(tle_line1[20:32])
    epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    return epoch.strftime('%Y-%m-%d %H:%M:%S')


def extract_tle1_features(tle_line1):
    # From line1: parse epoch and first derivative of mean motion (cols 34-43)
    epoch = parse_epoch_from_tle1(tle_line1)
    first_derivative_mean_motion = float(tle_line1[33:43].strip())
    return {
        'epoch': epoch,
        'first_derivative_mean_motion': first_derivative_mean_motion,
    }

def extract_tle2_features(tle_line2):
    # Parse key orbital parameters from line 2 (fixed width positions)
    return {
        'inclination_deg': float(tle_line2[8:16]),
        'raan_deg': float(tle_line2[17:25]),
        'eccentricity': float('0.' + tle_line2[26:33].strip()),
        'arg_perigee_deg': float(tle_line2[34:42]),
        'mean_anomaly_deg': float(tle_line2[43:51]),
        'mean_motion_revs_per_day': float(tle_line2[52:63]),
    }

# loading data CSV
tle_df = pd.read_csv('processed_satellite_data.csv')  # make sure tle1 and tle2 columns exist

# Extract features from tle1
tle1_features = tle_df['tle1'].apply(extract_tle1_features).apply(pd.Series)

# Extract features from tle2
tle2_features = tle_df['tle2'].apply(extract_tle2_features).apply(pd.Series)

# Combine all features into one DataFrame
tle_df = pd.concat([tle_df, tle1_features, tle2_features], axis=1)

# Check the result
print(tle_df.head())

# Save to CSV if needed
tle_df.to_csv('tle_with_parsed_features.csv', index=False)
