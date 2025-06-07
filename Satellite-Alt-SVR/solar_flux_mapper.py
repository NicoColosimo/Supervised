import pandas as pd
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- Load your Excel file ---
df = pd.read_excel("standardized_satellite_data.xlsx")

# --- Convert 'time' column to date only ---
df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %I:%M:%S %p').dt.date

# --- Fetch NOAA daily F10.7 solar flux data ---
url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
response = requests.get(url)
data = response.json()

# --- Create a dictionary mapping {date: flux} ---
flux_dict = {
    datetime.strptime(entry['time_tag'], "%Y-%m-%dT%H:%M:%S").date(): entry['flux']
    for entry in data
}

# --- Map solar flux to each satellite observation date ---
df['f107_solar_flux'] = df['time'].map(flux_dict)

# --- Drop rows where flux was not found ---
df.dropna(subset=['f107_solar_flux'], inplace=True)

# --- Standardize the flux column ---
scaler = StandardScaler()
df['f107_solar_flux'] = scaler.fit_transform(df[['f107_solar_flux']])

# --- Drop original date column if no longer needed ---
df.drop(columns=['time'], inplace=True)

# --- Save the updated dataset ---
df.to_excel("satellite_data_with_flux.xlsx", index=False)