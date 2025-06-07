import pandas as pd
import numpy as np
from math import degrees

# ========== CONFIG ========== #
FILE_PATH = "synthetic_gait_dataset.csv"
OUTPUT_PATH = "synthetic_gait_dataset_withID.csv"
FRAMES_PER_PERSON = 139
WINDOW_SIZE = 30
STEP = 10
# ============================ #

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

df = pd.read_csv(FILE_PATH)
num_people = len(df) // FRAMES_PER_PERSON
samples = []

for person_index in range(num_people):
    person_df = df.iloc[person_index * FRAMES_PER_PERSON : (person_index + 1) * FRAMES_PER_PERSON]
    person_id = f"person_{person_index+1:02d}"

    for start in range(0, FRAMES_PER_PERSON - WINDOW_SIZE + 1, STEP):
        segment = person_df.iloc[start:start + WINDOW_SIZE]
        segment_size = len(segment) // 5
        features = {}

        for i in range(5):
            seg = segment.iloc[i * segment_size:(i + 1) * segment_size]
            dists, angs, torso_leans, vels = [], [], [], []
            left_wrist_path = 0
            right_wrist_path = 0

            for j in range(1, len(seg)):
                row_prev = seg.iloc[j - 1]
                row = seg.iloc[j]

                def pt(name, r): return (r[f"{name} (x)"], r[f"{name} (y)"])

                joints = {name: pt(name, row) for name in [
                    "Left hip", "Right hip", "Left knee", "Right knee",
                    "Left ankle", "Right ankle", "Left shoulder", "Right shoulder",
                    "Left foot index", "Right foot index", "Left wrist", "Right wrist"
                ]}
                joints_prev = {name: pt(name, row_prev) for name in joints}

                dists.append({
                    "dist_hip_width": euclidean(joints["Left hip"], joints["Right hip"]),
                    "dist_knee_width": euclidean(joints["Left knee"], joints["Right knee"]),
                    "dist_ankle_width": euclidean(joints["Left ankle"], joints["Right ankle"]),
                    "dist_stride_length": euclidean(joints["Left foot index"], joints["Right foot index"]),
                    "dist_leg_len_L": euclidean(joints["Left hip"], joints["Left ankle"]),
                    "dist_leg_len_R": euclidean(joints["Right hip"], joints["Right ankle"]),
                    "dist_shoulder_width": euclidean(joints["Left shoulder"], joints["Right shoulder"])
                })

                torso_angle = angle_between(joints["Left shoulder"], joints["Left hip"], joints["Right hip"])
                torso_leans.append(torso_angle)

                angs.append({
                    "angle_knee_L": angle_between(joints["Left hip"], joints["Left knee"], joints["Left ankle"]),
                    "angle_knee_R": angle_between(joints["Right hip"], joints["Right knee"], joints["Right ankle"]),
                    "angle_hip_L": angle_between(joints["Left shoulder"], joints["Left hip"], joints["Left knee"]),
                    "angle_hip_R": angle_between(joints["Right shoulder"], joints["Right hip"], joints["Right knee"]),
                })

                vels.append({
                    "vel_l_ankle": euclidean(joints["Left ankle"], joints_prev["Left ankle"]),
                    "vel_r_ankle": euclidean(joints["Right ankle"], joints_prev["Right ankle"]),
                    "vel_l_knee": euclidean(joints["Left knee"], joints_prev["Left knee"]),
                    "vel_r_knee": euclidean(joints["Right knee"], joints_prev["Right knee"]),
                    "vel_l_wrist": euclidean(joints["Left wrist"], joints_prev["Left wrist"]),
                    "vel_r_wrist": euclidean(joints["Right wrist"], joints_prev["Right wrist"]),
                })

                left_wrist_path += euclidean(joints["Left wrist"], joints_prev["Left wrist"])
                right_wrist_path += euclidean(joints["Right wrist"], joints_prev["Right wrist"])

            prefix = f"seg{int(i*100/5)}"
            for k in dists[0]:
                features[f"{k}_{prefix}"] = np.mean([d[k] for d in dists])
            for k in angs[0]:
                features[f"{k}_{prefix}"] = np.mean([a[k] for a in angs])
            for k in vels[0]:
                features[f"{k}_{prefix}"] = np.mean([v[k] for v in vels])
            features[f"torso_lean_std_{prefix}"] = np.std(torso_leans)
            features[f"asym_arm_swing_{prefix}"] = abs(left_wrist_path - right_wrist_path)

        features["person_id"] = person_id
        features["frame_range"] = f"f_{start}-{start + WINDOW_SIZE - 1}"
        samples.append(features)

df_out = pd.DataFrame(samples)
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"Extracted {len(samples)} total samples across {num_people} people.")
print(f"Output saved to: {OUTPUT_PATH}")
