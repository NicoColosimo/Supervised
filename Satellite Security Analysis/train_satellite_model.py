import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import entropy
from catboost import CatBoostClassifier

# === Load Data ===
df = pd.read_csv("synthetic_satellite_data.csv")

# Pivot features by series_id and time_step
features = [col for col in df.columns if col not in ['class_label', 'series_id', 'time_step']]
pivot_dfs = []
for feat in features:
    pivot_df = df.pivot(index='series_id', columns='time_step', values=feat)
    pivot_df.columns = [f"{feat}_{int(col)}" for col in pivot_df.columns]
    pivot_dfs.append(pivot_df)

df_pivoted = pd.concat(pivot_dfs, axis=1)
labels = df.groupby('series_id')['class_label'].first()
df_pivoted['class_label'] = labels

X = df_pivoted.drop(columns=['class_label'])
y = df_pivoted['class_label']

# === Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA Visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 7))
for label, name in zip([0, 1, 2], ['Normal', 'System Issue', 'Compromised']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], alpha=0.6, label=f"{label}: {name}")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Satellite Telemetry (Satellite-level features)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Train-Test Split ===
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.3, random_state=42, stratify=y)

# === CatBoost Model ===
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    random_seed=42
)
cat_model.fit(X_train, y_train)

# === Evaluation ===
train_preds = cat_model.predict(X_train).ravel()
test_preds = cat_model.predict(X_test).ravel()
probs = cat_model.predict_proba(X_test)

print(f"\nTraining Accuracy: {accuracy_score(y_train, train_preds):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}\n")

print("Classification Report (Test):")
print(classification_report(y_test, test_preds, target_names=['Normal', 'System Issue', 'Compromised']))

print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, test_preds))

# === Cross-validation ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cat_model, X_scaled, y, cv=cv, scoring='accuracy')
print("\nCross-Validation Results (5-fold):")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: Accuracy = {score:.4f}")
print(f"  Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"  Std CV Accuracy: {np.std(cv_scores):.4f}")

# === Feature Importance ===
importances = cat_model.get_feature_importance()
sorted_idx = np.argsort(importances)[::-1]
top_n = 15  # change this if you want more/less

plt.figure(figsize=(10, 6))
plt.barh([X.columns[i] for i in sorted_idx[:top_n]][::-1],
         importances[sorted_idx[:top_n]][::-1], color='teal')
plt.xlabel("Importance")
plt.title("Top Feature Importances from CatBoost")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()

# === Threshold Adjustment for Compromised Class ===
threshold = 0.4
y_pred_adjusted = []

for prob_normal, prob_issue, prob_comp in probs:
    if prob_comp > threshold:
        y_pred_adjusted.append(2)
    else:
        y_pred_adjusted.append(0 if prob_normal > prob_issue else 1)

print(f"\nClassification Report (Test) with Compromised Threshold={threshold}:")
print(classification_report(y_test, y_pred_adjusted, target_names=['Normal', 'System Issue', 'Compromised']))

print("Confusion Matrix (Test) with Threshold Adjustment:")
print(confusion_matrix(y_test, y_pred_adjusted))

# === Entropy-based Uncertainty ===
uncertainty = entropy(np.array(probs).T)
uncertainty_threshold = 0.6
uncertain_samples = uncertainty > uncertainty_threshold

print(f"\nNumber of uncertain samples: {uncertain_samples.sum()} out of {len(uncertainty)}")
for idx in np.where(uncertain_samples)[0]:
    pred_class = y_pred_adjusted[idx]
    pred_prob = probs[idx][pred_class]
    print(f"Sample {idx}: Predicted class = {pred_class}, "
          f"Predicted probability = {pred_prob:.2f}, "
          f"Uncertainty (entropy) = {uncertainty[idx]:.2f}")


# === Feature Deviations for Uncertain Predictions ===
series_ids = df_pivoted.index.to_list()
uncertain_series_ids = [series_ids[test_idx[i]] for i in np.where(uncertain_samples)[0]]

X_train_unscaled = scaler.inverse_transform(X_train)
train_means = np.mean(X_train_unscaled, axis=0)
feature_names = X.columns.tolist()
X_test_unscaled = scaler.inverse_transform(X_test)

label_map = {0: "Normal", 1: "System Issue", 2: "Compromised"}

# === Check Samples Near Entropy ~0.55 ===
target_entropy = 0.55
tolerance = 0.01
close_to_threshold = np.where((uncertainty > target_entropy - tolerance) & (uncertainty < target_entropy + tolerance))[0]

print(f"\nSamples near entropy = {target_entropy} ± {tolerance}:")
for idx in close_to_threshold:
    pred_class = y_pred_adjusted[idx]
    pred_prob = probs[idx][pred_class]
    full_probs = probs[idx]
    print(f"Sample {idx}:")
    print(f"  Predicted class = {pred_class} ({label_map[pred_class]})")
    print(f"  Predicted probability = {pred_prob:.2f}")
    print(f"  Entropy = {uncertainty[idx]:.3f}")
    print(f"  Full class probabilities = {full_probs}\n")

with open("uncertain_satellites.txt", "w") as f:
    for idx_in_uncertain in np.where(uncertain_samples)[0]:
        sid = series_ids[test_idx[idx_in_uncertain]]
        f.write(f"Satellite Number: {sid}\n")

        sample_features = X_test_unscaled[idx_in_uncertain]
        deviations = np.abs(sample_features - train_means)
        top_idx = np.argsort(deviations)[-3:][::-1]

        f.write("Top 3 Features of Concern (highest deviation):\n")
        for i in top_idx:
            f.write(f"  - {feature_names[i]}: {sample_features[i]:.3f} (deviation: {deviations[i]:.3f})\n")

        pred_probs = probs[idx_in_uncertain]
        pred_class = y_pred_adjusted[idx_in_uncertain]
        pred_label = label_map[pred_class]
        alt_class = np.argsort(pred_probs)[-2]
        alt_label = label_map[alt_class]

        f.write(f"Predicted: \"{pred_label}\" but due to uncertainty can be \"{alt_label}\"\n\n")

print("\nUncertain satellite IDs and feature concerns saved to 'uncertain_satellites.txt'")

# === Plot Uncertainty Distribution ===
plt.figure(figsize=(8, 5))
plt.hist(uncertainty, bins=30, alpha=0.7, color='skyblue')
plt.axvline(uncertainty_threshold, color='red', linestyle='--', label='Uncertainty Threshold')
plt.title('Distribution of Prediction Uncertainty (Entropy)')
plt.xlabel('Entropy')
plt.ylabel('Number of Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
