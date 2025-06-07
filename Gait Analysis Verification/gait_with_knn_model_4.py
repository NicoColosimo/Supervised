import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report

# === Load data ===
df = pd.read_excel("synthetic_gait_data_11people_10samples_each.xlsx")

# === Split data into train/test (stratified by person_id) ===
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['person_id'], random_state=42)

# === Compute Distance Threshold from TRAIN data only ===
intra_dists = []
for person_id in np.unique(train_df['person_id']):
    person_samples = train_df[train_df['person_id'] == person_id].drop(columns='person_id').values
    dists = euclidean_distances(person_samples, person_samples)
    intra_dists += list(dists[np.triu_indices_from(dists, k=1)])

mean_dist = np.mean(intra_dists)
std_dist = np.std(intra_dists)
threshold = mean_dist + 2 * std_dist

print(f"Using distance threshold = {threshold:.4f}")

# === Verification function using TRAIN data as reference ===
def verify_knn_with_threshold(query, claimed_id, reference_df, k=3, threshold=threshold):
    claimed_samples = reference_df[reference_df['person_id'] == claimed_id].drop(columns=['person_id']).values
    dists = np.sort(euclidean_distances([query], claimed_samples)[0])[:k]
    avg_dist = np.mean(dists)
    return 1 if avg_dist <= threshold else 0, avg_dist

# === Run verification on TEST data ===
true_labels = []
predicted_labels = []

accepted_log = []
rejected_log = []

for person_id in test_df['person_id'].unique():
    person_test_samples = test_df[test_df['person_id'] == person_id]
    
    for idx, row in person_test_samples.iterrows():
        query_vec = row.drop('person_id').values

        # Genuine claim (correct ID)
        prediction, dist = verify_knn_with_threshold(query_vec, person_id, train_df, k=3)
        true_labels.append(1)
        predicted_labels.append(prediction)
        if prediction == 1:
            accepted_log.append(f"✔️ Accepted: True Person {person_id} claimed ID {person_id} | Dist = {dist:.4f}")
        else:
            rejected_log.append(f"❌ Rejected: True Person {person_id} claimed ID {person_id} | Dist = {dist:.4f}")

        # Impostor claims (all other IDs)
        for impostor_id in train_df['person_id'].unique():
            if impostor_id == person_id:
                continue
            prediction, dist = verify_knn_with_threshold(query_vec, impostor_id, train_df, k=3)
            true_labels.append(0)
            predicted_labels.append(prediction)
            if prediction == 1:
                accepted_log.append(f"❌ Wrongly Accepted: Person {person_id} falsely claimed ID {impostor_id} | Dist = {dist:.4f}")
            else:
                rejected_log.append(f"✔️ Correctly Rejected: Person {person_id} falsely claimed ID {impostor_id} | Dist = {dist:.4f}")

print("\n📊 Verification Report on TEST data:")
print(classification_report(true_labels, predicted_labels))

print(f"\n🗂️ Accepted Claims Log (first 10):")
for line in accepted_log[:10]:
    print(line)

print(f"\n🛡️ Rejected Claims Log (first 10):")
for line in rejected_log[:10]:
    print(line)
