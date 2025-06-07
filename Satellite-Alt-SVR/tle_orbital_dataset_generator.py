import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sgp4.api import Satrec, jday
import time

# ------------- Step 1: Download TLEs from Space-Track --------------

def fetch_tles_space_track(username, password, days=30):
    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    payload = {'identity': username, 'password': password}
    resp = session.post(login_url, data=payload)
    if resp.status_code != 200 or "login failed" in resp.text.lower():
        raise Exception("Login failed. Check your credentials.")
    date_cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    query_url = f"https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3E{date_cutoff}/orderby/NORAD_CAT_ID,EPOCH/format/3le"
    tle_resp = session.get(query_url)
    tle_resp.raise_for_status()
    return tle_resp.text

# ------------- Step 2: Parse TLEs --------------------

def parse_tles_3le(tle_text):
    lines = tle_text.strip().splitlines()
    sats = []
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        tle1 = lines[i+1].strip()
        tle2 = lines[i+2].strip()
        sats.append((name, tle1, tle2))
    return sats

def get_norad_id(tle_line_2):
    norad_str = tle_line_2[2:7]
    if not norad_str.isdigit():
        return None
    return int(norad_str)

def parse_epoch(tle_line_1):
    epoch_str = tle_line_1[18:32]
    year = int(epoch_str[:2])
    year += 2000 if year < 57 else 1900
    day_of_year = float(epoch_str[2:])
    dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    return dt

# ------------- Step 3: Compute orbital parameters ------------

def compute_orbital_params(tle1, tle2):
    sat = Satrec.twoline2rv(tle1, tle2)
    epoch_dt = parse_epoch(tle1)
    jd, fr = jday(epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute, epoch_dt.second + epoch_dt.microsecond*1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None
    altitude = np.linalg.norm(r) - 6371.0
    return {
        "epoch": epoch_dt,
        "altitude_km": altitude,
        "inclination_deg": np.degrees(sat.inclo),
        "eccentricity": sat.ecco,
        "semi_major_axis_km": sat.a,
        "perigee_km": sat.a * (1 - sat.ecco) - 6371.0,
        "apogee_km": sat.a * (1 + sat.ecco) - 6371.0,
    }

# ------------- Step 4: Fetch solar flux and Kp index -------------

def fetch_solar_geomag():
    try:
        f107_url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
        flux_data = requests.get(f107_url).json()
        latest_flux = next((float(entry["f107"]) for entry in reversed(flux_data) if entry["f107"] != -1), 70.0)
    except Exception:
        latest_flux = 70.0

    try:
        kp_url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        kp_data = requests.get(kp_url).json()
        kp_values = [float(e["kp_index"]) for e in kp_data[-24:] if e["kp_index"] not in (None, -1)]
        kp_avg = sum(kp_values) / len(kp_values) if kp_values else None
    except Exception:
        kp_avg = None

    return latest_flux, kp_avg

# ------------- Step 5: Build dataset ------------------

def build_dataset(tles):
    records = []
    for name, tle1, tle2 in tles:
        norad_id = get_norad_id(tle2)
        if norad_id is None:
            print(f"Skipping invalid TLE for satellite: {name}")
            continue
        orbital_params = compute_orbital_params(tle1, tle2)
        if orbital_params is None:
            print(f"Skipping due to sgp4 error: {name}")
            continue
        records.append({
            "norad_id": norad_id,
            "satellite_name": name,
            **orbital_params,
            "tle1": tle1,
            "tle2": tle2,
        })
    df = pd.DataFrame(records)
    return df

# ------------- Main routine -----------------------------

def main():
    username = "YOUR_USERNAME"
    password = "YOUR_PASSWORD"
    print("Downloading TLE data...")
    tle_text = fetch_tles_space_track(username, password)

    print("Parsing TLEs...")
    tles = parse_tles_3le(tle_text)

    print(f"Processing {len(tles)} TLE records to compute orbital parameters...")
    df = build_dataset(tles)

    print("Fetching solar and geomagnetic indices...")
    f107, kp = fetch_solar_geomag()
    print(f"Current F10.7 Solar Flux: {f107}")
    print(f"Current Kp index (average last 24 hours): {kp}")

    df["f107_solar_flux"] = f107
    df["kp_index"] = kp

    output_file = "satellite_orbit_decay_dataset.csv"
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
