import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# --- Setup ---
video_path = "media.mp4"
output_csv_prefix = "gait_view"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Constants for smoothing
FEET_LANDMARKS = {27, 28, 29, 30, 31, 32}
VISIBILITY_THRESHOLD = 0.5

# Smoothing parameters
ALPHA_NORMAL = 0.3
ALPHA_FEET_NORMAL = 0.1
ALPHA_FEET_FAST = 0.2  # Reduced for faster reactivity
VELOCITY_ALPHA = 0.6

# Thresholds for foot position freeze based on visibility
VISIBILITY_DROP_THRESHOLD = 0.3
VISIBILITY_RECOVER_THRESHOLD = 0.5

# Minimum velocity to consider "fast movement" of foot to increase smoothing speed
MIN_VELOCITY_TO_SMOOTH = 0.005
BIG_MOVE_THRESHOLD = 0.02
VELOCITY_UNSTICK_THRESHOLD = 0.002  # Small motion for frozen foot

# Initialize global vars for velocity smoothing & foot tracking
prev_velocity = [0.0] * (33 * 2)  # x,y velocities per landmark
last_stable_foot_pos = {i: None for i in FEET_LANDMARKS}
last_stable_visibility = {i: 1.0 for i in FEET_LANDMARKS}

# For smoothing visibility over a few frames per landmark
VISIBILITY_SMOOTH_FRAMES = 3
vis_history = {i: [1.0]*VISIBILITY_SMOOTH_FRAMES for i in range(33)}

def smooth_visibility(idx, curr_vis):
    # Simple moving average over last VISIBILITY_SMOOTH_FRAMES
    hist = vis_history[idx]
    hist.pop(0)
    hist.append(curr_vis)
    return sum(hist) / len(hist)

def prevent_foot_overlap(kp):
    # Adjusted for side view: prevent vertical overlap
    left_ankle_idx = 31
    right_ankle_idx = 32

    lx, ly = kp[left_ankle_idx * 3], kp[left_ankle_idx * 3 + 1]
    rx, ry = kp[right_ankle_idx * 3], kp[right_ankle_idx * 3 + 1]

    vertical_dist = abs(ly - ry)
    min_vertical_dist = 0.015  # Tighter since it's y-axis in side view

    if vertical_dist < min_vertical_dist:
        mid_y = (ly + ry) / 2
        kp[left_ankle_idx * 3 + 1] = mid_y - min_vertical_dist / 2
        kp[right_ankle_idx * 3 + 1] = mid_y + min_vertical_dist / 2

    return kp

def smooth_keypoints_with_visibility_and_velocity(prev_kp, curr_kp):
    global prev_velocity, last_stable_foot_pos, last_stable_visibility

    if prev_kp is None:
        # Initialize velocity and foot position tracking on first frame
        prev_velocity = [0.0] * (33 * 2)
        for i in range(33):
            vis_history[i] = [curr_kp[i*3 + 2]] * VISIBILITY_SMOOTH_FRAMES
        for i in FEET_LANDMARKS:
            last_stable_foot_pos[i] = curr_kp[i*3:i*3+3]
            last_stable_visibility[i] = curr_kp[i*3 + 2]
        return curr_kp

    smoothed = []
    for i in range(33):
        base_idx = i * 3
        x_prev, y_prev = prev_kp[base_idx], prev_kp[base_idx + 1]
        x_curr, y_curr = curr_kp[base_idx], curr_kp[base_idx + 1]
        vis_curr_raw = curr_kp[base_idx + 2]

        vis_curr = smooth_visibility(i, vis_curr_raw)

        v_idx = i * 2
        vx = x_curr - x_prev
        vy = y_curr - y_prev

        vx_smooth = VELOCITY_ALPHA * vx + (1 - VELOCITY_ALPHA) * prev_velocity[v_idx]
        vy_smooth = VELOCITY_ALPHA * vy + (1 - VELOCITY_ALPHA) * prev_velocity[v_idx + 1]

        prev_velocity[v_idx] = vx_smooth
        prev_velocity[v_idx + 1] = vy_smooth

        velocity_mag = np.sqrt(vx_smooth ** 2 + vy_smooth ** 2)

        if i in FEET_LANDMARKS:
            if vis_curr < VISIBILITY_DROP_THRESHOLD:
                if velocity_mag > VELOCITY_UNSTICK_THRESHOLD:
                    alpha = ALPHA_FEET_FAST
                else:
                    pos = last_stable_foot_pos[i]
                    if pos is not None:
                        smoothed.extend(pos)
                    else:
                        smoothed.extend(curr_kp[base_idx:base_idx+3])
                    continue
            elif vis_curr >= VISIBILITY_RECOVER_THRESHOLD:
                last_stable_foot_pos[i] = [x_curr, y_curr, vis_curr]
                last_stable_visibility[i] = vis_curr

            if velocity_mag < MIN_VELOCITY_TO_SMOOTH:
                alpha = 1.0  # No smoothing for very slow/steady movement
            else:
                alpha = ALPHA_FEET_FAST if velocity_mag > BIG_MOVE_THRESHOLD else ALPHA_FEET_NORMAL
        else:
            alpha = ALPHA_NORMAL

        # Smooth x
        x_val = alpha * x_curr + (1 - alpha) * x_prev

        # Smooth y with additional damping for ankles
        if i in [31, 32]:  # Ankles
            damp_factor = 0.85
            y_val = damp_factor * y_prev + (1 - damp_factor) * y_curr
        else:
            y_val = alpha * y_curr + (1 - alpha) * y_prev

        # Smooth visibility
        vis_val = alpha * vis_curr + (1 - alpha) * prev_kp[base_idx + 2]

        smoothed.extend([x_val, y_val, vis_val])

    # Prevent feet overlap after smoothing
    smoothed = prevent_foot_overlap(smoothed)
    return smoothed

def keypoints_to_landmark_list(keypoints):
    from mediapipe.framework.formats import landmark_pb2
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for i in range(33):
        lm = landmark_list.landmark.add()
        lm.x = keypoints[i*3]
        lm.y = keypoints[i*3 + 1]
        lm.visibility = keypoints[i*3 + 2]
    return landmark_list

def draw_smoothed_keypoints(image, keypoints):
    landmark_list = keypoints_to_landmark_list(keypoints)
    mp_drawing.draw_landmarks(image, landmark_list, mp_pose.POSE_CONNECTIONS)

# --- Video Processing ---
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
midpoint = frame_width // 2

data_side = []
data_front = []

prev_side_kp = None
prev_front_kp = None
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    side_view = frame[:, :midpoint]
    front_view = frame[:, midpoint:]

    side_rgb = cv2.cvtColor(side_view, cv2.COLOR_BGR2RGB)
    front_rgb = cv2.cvtColor(front_view, cv2.COLOR_BGR2RGB)

    # Process side view
    side_results = pose.process(side_rgb)
    curr_side_kp = None
    if side_results.pose_landmarks:
        curr_side_kp = []
        for lm in side_results.pose_landmarks.landmark:
            curr_side_kp.extend([lm.x, lm.y, lm.visibility])

    # Process front view
    front_results = pose.process(front_rgb)
    curr_front_kp = None
    if front_results.pose_landmarks:
        curr_front_kp = []
        for lm in front_results.pose_landmarks.landmark:
            curr_front_kp.extend([lm.x, lm.y, lm.visibility])

    # Smooth keypoints if detected
    if curr_side_kp:
        smoothed_side = smooth_keypoints_with_visibility_and_velocity(prev_side_kp, curr_side_kp)
        prev_side_kp = smoothed_side
        data_side.append([frame_id] + smoothed_side)
        draw_smoothed_keypoints(side_view, smoothed_side)
    else:
        smoothed_side = prev_side_kp  # Fallback

    if curr_front_kp:
        smoothed_front = smooth_keypoints_with_visibility_and_velocity(prev_front_kp, curr_front_kp)
        prev_front_kp = smoothed_front
        data_front.append([frame_id] + smoothed_front)
        draw_smoothed_keypoints(front_view, smoothed_front)
    else:
        smoothed_front = prev_front_kp  # Fallback

    # Combine side + front for display
    combined = cv2.hconcat([side_view, front_view])
    cv2.imshow("Gait Keypoints (Side & Front)", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
pose.close()

# --- Save CSV ---
columns = ['frame']
for i in range(33):
    columns += [f'x_{i}', f'y_{i}', f'vis_{i}']

pd.DataFrame(data_side, columns=columns).to_csv(f"{output_csv_prefix}_side.csv", index=False)
pd.DataFrame(data_front, columns=columns).to_csv(f"{output_csv_prefix}_front.csv", index=False)

print(f"Keypoints extracted and saved:")
print(f"- Side view keypoints to {output_csv_prefix}_side.csv")
print(f"- Front view keypoints to {output_csv_prefix}_front.csv")
