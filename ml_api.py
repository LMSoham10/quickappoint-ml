from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI(title="QuickAppoint EWT Prediction Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("quickappoint_rf_model.joblib")

DOCTOR_BASE_PACE = {
    "DOC_101": 8.0,
    "DOC_102": 10.5,
    "DOC_201": 16.0,
    "DOC_301": 11.0,
    "DOC_401": 13.5,
}

SPECIALIZATION_DEFAULT_PACE = {
    "General Medicine": 9.5,
    "Cardiology": 16.0,
    "Pediatrics": 11.0,
    "Orthopedics": 13.5,
}

PROBLEM_WEIGHTS = {
    "Prescription Refill": -3.5,
    "Routine Fever/Cold": 0.0,
    "Pediatric Vaccination": -2.0,
    "Joint/Back Pain": 3.0,
    "Chest Pain / ECG": 7.5,
    "Chronic Diabetes Review": 2.5,
}

class PatientQueueItem(BaseModel):
    doctor_id: str
    doctor_specialization: str
    type_of_patient: str
    common_problem: str
    position_X3: int
    visit_type_X4: str
    day_of_week_X5: str
    hist_avg_X1: Optional[float] = None
    recent_pace_X2: Optional[float] = None

class QueueRequest(BaseModel):
    upcoming_queue: List[PatientQueueItem]
    delay_offset_mins: float = 0.0
    live_shift_avg_mins: Optional[float] = None

def enrich_features(item: PatientQueueItem, live_shift_avg: Optional[float]) -> dict:
    # 1. Auto-calculate hist_avg_X1 if not provided
    if item.hist_avg_X1 is not None:
        hist_avg = item.hist_avg_X1
    else:
        base = DOCTOR_BASE_PACE.get(
            item.doctor_id,
            SPECIALIZATION_DEFAULT_PACE.get(item.doctor_specialization, 11.0)
        )
        hist_avg = base + PROBLEM_WEIGHTS.get(item.common_problem, 0.0)
        if item.type_of_patient == "Geriatric":
            hist_avg += 2.0
        if item.visit_type_X4 == "New":
            hist_avg += 2.5
        hist_avg = max(4.0, round(hist_avg, 2))

    # 2. Auto-calculate recent_pace_X2 from live shift history (or rush-hour estimate)
    if item.recent_pace_X2 is not None:
        recent_pace = item.recent_pace_X2
    elif live_shift_avg is not None:
        recent_pace = round(live_shift_avg, 2)
    else:
        day_factor = 1.12 if item.day_of_week_X5 in ["Monday", "Saturday"] else 1.0
        recent_pace = round(hist_avg * day_factor, 2)

    return {
        "doctor_id": item.doctor_id,
        "doctor_specialization": item.doctor_specialization,
        "type_of_patient": item.type_of_patient,
        "common_problem": item.common_problem,
        "hist_avg_X1": hist_avg,
        "recent_pace_X2": recent_pace,
        "position_X3": item.position_X3,
        "visit_type_X4": item.visit_type_X4,
        "day_of_week_X5": item.day_of_week_X5,
    }

@app.post("/predict-ewt")
def predict_ewt(payload: QueueRequest):
    if not payload.upcoming_queue:
        return {
            "doctor_target_benchmark_mins": 0.0,
            "total_ewt_mins": round(payload.delay_offset_mins, 1),
            "expected_arrival_clock": (datetime.now() + timedelta(minutes=payload.delay_offset_mins)).strftime("%I:%M %p"),
            "per_patient_durations": []
        }

    enriched_rows = [enrich_features(item, payload.live_shift_avg_mins) for item in payload.upcoming_queue]
    df = pd.DataFrame(enriched_rows)
    preds = model.predict(df)

    total_ewt = round(float(preds.sum()) + payload.delay_offset_mins, 1)
    arrival_clock = (datetime.now() + timedelta(minutes=total_ewt)).strftime("%I:%M %p")

    return {
        "doctor_target_benchmark_mins": round(float(preds[0]), 1),
        "total_ewt_mins": total_ewt,
        "expected_arrival_clock": arrival_clock,
        "per_patient_durations": [round(float(p), 1) for p in preds]
    }