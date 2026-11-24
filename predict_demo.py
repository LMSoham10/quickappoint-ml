import joblib
import pandas as pd
from datetime import datetime, timedelta

# Load the trained Random Forest pipeline
model = joblib.load("quickappoint_rf_model.joblib")

# Simulate 3 patients currently ahead of Token #4 in Cardiology (DOC_201)
upcoming_queue = pd.DataFrame([
    {
        "doctor_id": "DOC_201",
        "doctor_specialization": "Cardiology",
        "type_of_patient": "Adult",
        "common_problem": "Prescription Refill",
        "hist_avg_X1": 12.5,
        "recent_pace_X2": 13.0,
        "position_X3": 1,
        "visit_type_X4": "Follow-Up",
        "day_of_week_X5": "Monday",
    },
    {
        "doctor_id": "DOC_201",
        "doctor_specialization": "Cardiology",
        "type_of_patient": "Geriatric",
        "common_problem": "Chest Pain / ECG",
        "hist_avg_X1": 28.0,
        "recent_pace_X2": 30.5,
        "position_X3": 2,
        "visit_type_X4": "New",
        "day_of_week_X5": "Monday",
    },
    {
        "doctor_id": "DOC_201",
        "doctor_specialization": "Cardiology",
        "type_of_patient": "Chronic",
        "common_problem": "Chronic Diabetes Review",
        "hist_avg_X1": 18.5,
        "recent_pace_X2": 19.8,
        "position_X3": 3,
        "visit_type_X4": "Follow-Up",
        "day_of_week_X5": "Monday",
    },
])

# Predict consultation durations for each ahead token
predicted_durations = model.predict(upcoming_queue)
total_ewt_mins = round(float(predicted_durations.sum()), 1)
expected_arrival_clock = (datetime.now() + timedelta(minutes=total_ewt_mins)).strftime("%I:%M %p")

print(f"Doctor Target Benchmark (Current Patient): {predicted_durations[0]:.1f} mins")
print(f"Token #4 Expected Waiting Time (EWT): {total_ewt_mins} mins")
print(f"Expected Clock Time of Turn: {expected_arrival_clock}")