import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# === Load Data === #
df = pd.read_excel("Mouse_with_interaction_final.xlsx")

# === Feature and Target Selection === #
features_all = [
    'log_exposure_duration', 'ADR GCR (mGy/d)', 'ADR SAA (mGy/d)', 'ADR Total (mGy/d)',
    'TAD GCR (mGy)', 'TAD SAA (mGy)', 'TAD Total (mGy)', 'avg_daily_total_dose',
    'log_exposure_duration_x_TAD_Total', 'ADR_Total_x_TAD_GCR', 'TAD_GCR_x_TAD_SAA',
    'log_exposure_duration_x_TAD_SAA', 'TAD_GCR_x_TAD_Total', 'TAD_Total_x_avg_daily_total_dose'
]

features_base = [
    'log_exposure_duration', 'ADR GCR (mGy/d)', 'ADR SAA (mGy/d)', 'ADR Total (mGy/d)',
    'TAD GCR (mGy)', 'TAD SAA (mGy)', 'TAD Total (mGy)', 'avg_daily_total_dose'
]

targets = ['cns_dose_mGy', 'skin_dose_mGy', 'bfo_dose_mGy']

X_all = df[features_all]
X_base = df[features_base]
y = df[targets]

# === Train-Test Split === #
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
X_train_base, X_test_base, _, _ = train_test_split(X_base, y, test_size=0.2, random_state=42)

# === Ridge with Standardization === #
ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
multi_ridge_all = MultiOutputRegressor(ridge_pipeline)
multi_ridge_base = MultiOutputRegressor(ridge_pipeline)

multi_ridge_all.fit(X_train_all, y_train)
multi_ridge_base.fit(X_train_base, y_train)

y_pred_all = multi_ridge_all.predict(X_test_all)
y_pred_base = multi_ridge_base.predict(X_test_base)

# === Evaluation Function === #
def evaluate(y_true, y_pred, label):
    print(f"--- {label} ---")
    for i, target_name in enumerate(targets):
        mse = mean_squared_error(y_true.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
        print(f"{target_name} - MSE: {mse:.6f}, R^2: {r2:.6f}")
    print()

evaluate(y_test, y_pred_all, "All Features")
evaluate(y_test, y_pred_base, "Base Features Only")

# === K-Fold Cross-Validation === #
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse_scores = []
cv_r2_scores = []

for train_index, test_index in kf.split(X_all):
    X_train_cv, X_test_cv = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    model_cv = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)

    mse_fold = [mean_squared_error(y_test_cv.iloc[:, i], y_pred_cv[:, i]) for i in range(y.shape[1])]
    r2_fold = [r2_score(y_test_cv.iloc[:, i], y_pred_cv[:, i]) for i in range(y.shape[1])]

    cv_mse_scores.append(mse_fold)
    cv_r2_scores.append(r2_fold)

cv_mse_scores = np.array(cv_mse_scores)
cv_r2_scores = np.array(cv_r2_scores)

print("--- Cross-Validation Scores (All Features) ---")
for i, target_name in enumerate(targets):
    print(f"{target_name} - Mean CV MSE: {cv_mse_scores[:, i].mean():.6f}, Mean CV R^2: {cv_r2_scores[:, i].mean():.6f}")
print()

# === Ridge Alpha Tuning === #
alphas = [0.01, 0.1, 1.0, 10, 100]
print("--- Ridge alpha tuning ---")
for a in alphas:
    model = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=a)))
    model.fit(X_train_all, y_train)
    y_pred_alpha = model.predict(X_test_all)
    print(f"Alpha = {a}")
    for i, target_name in enumerate(targets):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred_alpha[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred_alpha[:, i])
        print(f"  {target_name} - MSE: {mse:.6f}, R^2: {r2:.6f}")
    print()

# === Plots: Predicted vs True === #
for i, target_name in enumerate(targets):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.iloc[:, i], y_pred_all[:, i], alpha=0.7)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
             [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--')
    plt.xlabel("True " + target_name)
    plt.ylabel("Predicted " + target_name)
    plt.title(f"Predicted vs True: {target_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Residual Plots === #
for i, target_name in enumerate(targets):
    residuals = y_test.iloc[:, i] - y_pred_all[:, i]
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred_all[:, i], residuals, alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted " + target_name)
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted: {target_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
