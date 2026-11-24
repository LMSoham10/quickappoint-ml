import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# 1. Load dataset & pre-trained Random Forest pipeline
df = pd.read_csv("quickappoint_demo_dataset.csv")
model = joblib.load("quickappoint_rf_model.joblib")

FEATURES = [
    "doctor_id",
    "doctor_specialization",
    "type_of_patient",
    "common_problem",
    "hist_avg_X1",
    "recent_pace_X2",
    "position_X3",
    "visit_type_X4",
    "day_of_week_X5",
]
TARGET = "actual_duration_mins"

X = df[FEATURES]
y = df[TARGET]

# Same 80/20 split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Generate Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 3. Regression Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
within_2_mins = np.mean(np.abs(y_test - y_test_pred) <= 2.0) * 100

print("=" * 60)
print("1. REGRESSION PERFORMANCE (CONTINUOUS DURATION IN MINS)")
print("=" * 60)
print(f"Train MAE                 : {train_mae:.3f} mins")
print(f"Test MAE                  : {test_mae:.3f} mins")
print(f"Test RMSE                 : {test_rmse:.3f} mins")
print(f"R² Score (Variance Expl.) : {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"Tolerance Accuracy (±2m)  : {within_2_mins:.2f}% of predictions within ±2 mins")

# 4. Sample Predictions Comparison Table
sample_df = X_test[["doctor_specialization", "type_of_patient", "common_problem", "visit_type_X4"]].head(8).copy()
sample_df["Actual (min)"] = np.round(y_test.head(8).values, 1)
sample_df["Predicted (min)"] = np.round(y_test_pred[:8], 1)
sample_df["Abs Error (min)"] = np.round(np.abs(sample_df["Actual (min)"] - sample_df["Predicted (min)"]), 2)

print("\n" + "=" * 60)
print("2. SAMPLE TEST SET PREDICTIONS VS ACTUAL")
print("=" * 60)
print(sample_df.to_string(index=False))

# 5. Classification Metrics (Bucketed Duration Tiers for Precision/Recall)
def categorize_duration(mins):
    if mins <= 12.0:
        return "Short (<=12m)"
    elif mins <= 20.0:
        return "Medium (12-20m)"
    else:
        return "Long (>20m)"

labels = ["Short (<=12m)", "Medium (12-20m)", "Long (>20m)"]
y_test_cls = [categorize_duration(v) for v in y_test]
y_pred_cls = [categorize_duration(v) for v in y_test_pred]

prec = precision_score(y_test_cls, y_pred_cls, average="weighted")
rec = recall_score(y_test_cls, y_pred_cls, average="weighted")
f1 = f1_score(y_test_cls, y_pred_cls, average="weighted")
cls_acc = accuracy_score(y_test_cls, y_pred_cls)

print("\n" + "=" * 60)
print("3. TIERED CLASSIFICATION METRICS (PRECISION, RECALL, F1)")
print("=" * 60)
print(f"Tier Classification Accuracy : {cls_acc*100:.2f}%")
print(f"Weighted Precision           : {prec:.4f}")
print(f"Weighted Recall              : {rec:.4f}")
print(f"Weighted F1-Score            : {f1:.4f}\n")
print(classification_report(y_test_cls, y_pred_cls, target_names=labels))

# 6. Generate 4-Panel Evaluation Graph
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"QuickAppoint Random Forest EWT Model Evaluation (MAE: {test_mae:.2f}m | RMSE: {test_rmse:.2f}m | R²: {test_r2:.3f})",
    fontsize=14,
    fontweight="bold",
)

# Panel A: Actual vs Predicted Scatter
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color="#2563eb", edgecolors="k", linewidth=0.4)
min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
axes[0, 0].set_title("Actual vs. Predicted Consultation Duration")
axes[0, 0].set_xlabel("Actual Duration (minutes)")
axes[0, 0].set_ylabel("Predicted Duration (minutes)")
axes[0, 0].legend()

# Panel B: Residual Error Distribution
residuals = y_test - y_test_pred
sns.histplot(residuals, bins=25, kde=True, color="#10b981", ax=axes[0, 1])
axes[0, 1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[0, 1].set_title("Prediction Error (Residual) Distribution")
axes[0, 1].set_xlabel("Residual Error: Actual - Predicted (minutes)")
axes[0, 1].set_ylabel("Patient Count")

# Panel C: Confusion Matrix (Duration Tiers)
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[1, 0],
)
axes[1, 0].set_title("Confusion Matrix (Consultation Duration Tiers)")
axes[1, 0].set_xlabel("Predicted Tier")
axes[1, 0].set_ylabel("Actual Tier")

# Panel D: Top Feature Importances
preprocessor = model.named_steps["preprocessor"]
rf_estimator = model.named_steps["regressor"]
cat_cols = preprocessor.transformers_[0][2]
num_cols = preprocessor.transformers_[1][2]
ohe_features = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([ohe_features, num_cols])

importances = pd.Series(rf_estimator.feature_importances_, index=all_feature_names)
top_10 = importances.sort_values(ascending=True).tail(10)
top_10.plot(kind="barh", color="#6366f1", ax=axes[1, 1])
axes[1, 1].set_title("Top 10 Most Influential Features")
axes[1, 1].set_xlabel("Relative Importance Score")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("model_evaluation_report.png", dpi=300)
print("Saved high-resolution 4-panel chart to: model_evaluation_report.png")
plt.show()