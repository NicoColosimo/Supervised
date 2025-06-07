import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import skew

# --- Load the normalized dataset ---
df = pd.read_excel("updated_flux_data.xlsx")

# --- Define features and target ---
target = 'altitude_km'
X = df.drop(columns=[target])
y = df[target]

# --- Analyze target distribution ---
skewness_transformed = skew(y)
print(f"Skewness of transformed altitude_km: {skewness_transformed:.4f}")

# --- Plot histogram ---
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram of Transformed altitude_km (Skewness={skewness_transformed:.4f})')
plt.xlabel('Transformed altitude_km')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("histogram_transformed_altitude.png")
plt.show()

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train SVR model ---
svr = SVR(kernel='rbf', C=10, epsilon=0.1, gamma=0.1)
svr.fit(X_train, y_train)

# --- Predictions ---
y_pred = svr.predict(X_test)

# --- Evaluation metrics ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# --- 1. Predicted vs Actual Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='dodgerblue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Altitude (Normalized)")
plt.ylabel("Predicted Altitude (Normalized)")
plt.title("SVR: Predicted vs Actual Altitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()

# --- 2. Residual Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6, color='orange', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Altitude")
plt.ylabel("Residuals")
plt.title("SVR Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()

# --- 3. Feature Importance (Permutation) ---
result = permutation_importance(svr, X_test, y_test, n_repeats=10, random_state=42)
importance = pd.Series(result.importances_mean, index=X.columns)

plt.figure(figsize=(8, 6))
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Feature Importance (Permutation)")
plt.xlabel("Mean Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
