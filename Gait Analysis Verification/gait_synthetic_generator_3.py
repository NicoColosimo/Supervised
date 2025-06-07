import numpy as np
import pandas as pd

# Your original vector
# This is a sample vector representing gait features for one person.
original = np.array(#Data is hidden for privacy reasons)

columns = [
    "hip_width_0_f", "knee_width_0_f", "ankle_width_0_f", "stride_length_0_f", "leg_len_L_0_f", "leg_len_R_0_f",
    "shoulder_width_0_f", "knee_angle_L_0_f", "knee_angle_R_0_f", "hip_angle_L_0_f", "hip_angle_R_0_f",
    "vel_l_ankle_0_f", "vel_r_ankle_0_f", "vel_l_knee_0_f", "vel_r_knee_0_f", "vel_l_wrist_0_f", "vel_r_wrist_0_f",
    "torso_lean_std_0_f", "arm_swing_asym_0_f", "hip_width_20_f", "knee_width_20_f", "ankle_width_20_f",
    "stride_length_20_f", "leg_len_L_20_f", "leg_len_R_20_f", "shoulder_width_20_f", "knee_angle_L_20_f",
    "knee_angle_R_20_f", "hip_angle_L_20_f", "hip_angle_R_20_f", "vel_l_ankle_20_f", "vel_r_ankle_20_f",
    "vel_l_knee_20_f", "vel_r_knee_20_f", "vel_l_wrist_20_f", "vel_r_wrist_20_f", "torso_lean_std_20_f",
    "arm_swing_asym_20_f", "hip_width_40_f", "knee_width_40_f", "ankle_width_40_f", "stride_length_40_f",
    "leg_len_L_40_f", "leg_len_R_40_f", "shoulder_width_40_f", "knee_angle_L_40_f", "knee_angle_R_40_f",
    "hip_angle_L_40_f", "hip_angle_R_40_f", "vel_l_ankle_40_f", "vel_r_ankle_40_f", "vel_l_knee_40_f",
    "vel_r_knee_40_f", "vel_l_wrist_40_f", "vel_r_wrist_40_f", "torso_lean_std_40_f", "arm_swing_asym_40_f",
    "hip_width_60_f", "knee_width_60_f", "ankle_width_60_f", "stride_length_60_f", "leg_len_L_60_f",
    "leg_len_R_60_f", "shoulder_width_60_f", "knee_angle_L_60_f", "knee_angle_R_60_f", "hip_angle_L_60_f",
    "hip_angle_R_60_f", "vel_l_ankle_60_f", "vel_r_ankle_60_f", "vel_l_knee_60_f", "vel_r_knee_60_f",
    "vel_l_wrist_60_f", "vel_r_wrist_60_f", "torso_lean_std_60_f", "arm_swing_asym_60_f", "hip_width_80_f",
    "knee_width_80_f", "ankle_width_80_f", "stride_length_80_f", "leg_len_L_80_f", "leg_len_R_80_f",
    "shoulder_width_80_f", "knee_angle_L_80_f", "knee_angle_R_80_f", "hip_angle_L_80_f", "hip_angle_R_80_f",
    "vel_l_ankle_80_f", "vel_r_ankle_80_f", "vel_l_knee_80_f", "vel_r_knee_80_f", "vel_l_wrist_80_f",
    "vel_r_wrist_80_f", "torso_lean_std_80_f", "arm_swing_asym_80_f"
]

def generate_sample(base, bounds, noise_factor):
    noisy = base + np.random.normal(0, noise_factor * np.abs(base))
    # Clip within bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    clipped = np.minimum(np.maximum(noisy, lower), upper)
    return clipped

# For one person, generate the 10 samples
def generate_person_samples(original_vector, n_samples=10):
    # Step 1: Generate first sample within ±30% bounds of original
    bounds_30 = []
    for val in original_vector:
        low = val * 0.7 if abs(val) > 1e-4 else val - 0.01
        high = val * 1.3 if abs(val) > 1e-4 else val + 0.01
        bounds_30.append((low, high))
    
    first_sample = generate_sample(original_vector, bounds_30, noise_factor=0.12)  # 12% noise approx inside 30% range
    
    # Step 2: Generate next n_samples-1 samples within ±1% bounds of first_sample
    bounds_1 = []
    for val in first_sample:
        low = val * 0.99 if abs(val) > 1e-4 else val - 0.0001
        high = val * 1.01 if abs(val) > 1e-4 else val + 0.0001
        bounds_1.append((low, high))
    
    samples = [first_sample]
    for _ in range(n_samples - 1):
        sample = generate_sample(first_sample, bounds_1, noise_factor=0.005)  # very tight noise
        samples.append(sample)
    
    return np.array(samples)

# Example: Generate data for 11 people, assuming the same original for demo
all_samples = []
all_person_ids = []

num_people = 11
samples_per_person = 10

for person_id in range(num_people):
    person_samples = generate_person_samples(original, samples_per_person)
    all_samples.append(person_samples)
    all_person_ids.extend([person_id] * samples_per_person)

all_samples = np.vstack(all_samples)

df = pd.DataFrame(all_samples, columns=columns)
df['person_id'] = all_person_ids

# Save to Excel
df.to_excel('synthetic_gait_data_11people_10samples_each.xlsx', index=False)
print("Data saved with shape:", df.shape)
