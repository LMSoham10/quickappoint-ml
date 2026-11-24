import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

np.random.seed(42)
n_samples = 1500

# 1. Define Doctor & Specialization Profiles
doctors = [
    {"doctor_id": "DOC_101", "specialization": "General Medicine", "base_pace": 8.0},
    {"doctor_id": "DOC_102", "specialization": "General Medicine", "base_pace": 10.5},
    {"doctor_id": "DOC_201", "specialization": "Cardiology", "base_pace": 16.0},
    {"doctor_id": "DOC_301", "specialization": "Pediatrics", "base_pace": 11.0},
    {"doctor_id": "DOC_401", "specialization": "Orthopedics", "base_pace": 13.5},
]

# Problem weights (in minutes)
problem_weights = {
    "Prescription Refill": -3.5,
    "Routine Fever/Cold": 0.0,
    "Pediatric Vaccination": -2.0,
    "Joint/Back Pain": 3.0,
    "Chest Pain / ECG": 7.5,
    "Chronic Diabetes Review": 2.5,
}

patient_types = ["Adult", "Pediatric", "Geriatric", "Chronic"]
visit_types = ["New", "Follow-Up"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

records = []
for _ in range(n_samples):
    doc = np.random.choice(doctors)
    p_type = np.random.choice(patient_types, p=[0.45, 0.20, 0.20, 0.15])
    problem = np.random.choice(list(problem_weights.keys()))
    v_type = np.random.choice(visit_types, p=[0.4, 0.6])
    day = np.random.choice(days_of_week)
    position = np.random.randint(1, 25)

    # Calculate Doctor + Problem Historical Average (X1)
    hist_avg = doc["base_pace"] + problem_weights[problem]
    if p_type == "Geriatric":
        hist_avg += 2.0
    if v_type == "New":
        hist_avg += 2.5
    hist_avg = max(4.0, round(hist_avg, 2))

    # Simulate Live Session Velocity / Recent Pace (X2)
    day_factor = 1.12 if day in ["Monday", "Saturday"] else 1.0
    recent_pace = round(hist_avg * day_factor * np.random.uniform(0.85, 1.25), 2)

    # Ground-Truth Target: actual_duration_mins
    actual_duration = (
        0.55 * hist_avg
        + 0.35 * recent_pace
        + (0.08 * position)
        + np.random.normal(0, 1.2)
    )
    actual_duration = max(3.0, round(actual_duration, 1))

    records.append({
        "doctor_id": doc["doctor_id"],
        "doctor_specialization": doc["specialization"],
        "type_of_patient": p_type,
        "common_problem": problem,
        "hist_avg_X1": hist_avg,
        "recent_pace_X2": recent_pace,
        "position_X3": position,
        "visit_type_X4": v_type,
        "day_of_week_X5": day,
        "actual_duration_mins": actual_duration,
    })

df = pd.DataFrame(records)
df.to_csv("quickappoint_demo_dataset.csv", index=False)
print("Saved quickappoint_demo_dataset.csv with", len(df), "rows.")

# 2. Train Random Forest Model
X = df.drop(columns=["actual_duration_mins"])
y = df["actual_duration_mins"]

categorical_cols = [
    "doctor_id",
    "doctor_specialization",
    "type_of_patient",
    "common_problem",
    "visit_type_X4",
    "day_of_week_X5",
]
numeric_cols = ["hist_avg_X1", "recent_pace_X2", "position_X3"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate Performance
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Model Trained Successfully > MAE: {mae:.2f} mins | RMSE: {rmse:.2f} mins")

# Save Model for Demo Inference
joblib.dump(model, "quickappoint_rf_model.joblib")
print("Saved quickappoint_rf_model.joblib")