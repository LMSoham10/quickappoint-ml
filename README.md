# QuickAppoint ML Service (FastAPI)

Random Forest Regressor microservice for predicting patient Expected Waiting Time (EWT) and continuous consultation logging.

## Local Setup & Run
1. Create and activate a virtual environment:
   \\\ash
   python -m venv .venv
   .\.venv\Scripts\activate      # Windows
   source .venv/bin/activate     # Mac/Linux
   \\\
2. Install dependencies:
   \\\ash
   pip install -r requirements.txt
   \\\
3. Start the FastAPI server:
   \\\ash
   uvicorn ml_api:app --host 0.0.0.0 --port 8000 --reload
   \\\
4. Interactive Swagger UI Docs: http://127.0.0.1:8000/docs

## Production Deployment Command (Render / Railway / VPS)
\\\ash
uvicorn ml_api:app --host 0.0.0.0 --port \
\\\

## API Endpoints

### 1. Predict Waiting Time (POST /predict-ewt)
Send the list of waiting patients ahead in the queue (upcoming_queue). hist_avg_X1 and ecent_pace_X2 are auto-calculated if omitted.

**Request:**
\\\json
{
  "delay_offset_mins": 0,
  "upcoming_queue": [
    {
      "doctor_id": "DOC_201",
      "doctor_specialization": "Cardiology",
      "type_of_patient": "Geriatric",
      "common_problem": "Chest Pain / ECG",
      "position_X3": 1,
      "visit_type_X4": "New",
      "day_of_week_X5": "Monday"
    }
  ]
}
\\\

**Response:**
\\\json
{
  "doctor_target_benchmark_mins": 24.6,
  "total_ewt_mins": 24.6,
  "expected_arrival_clock": "04:40 PM",
  "per_patient_durations": [24.6]
}
\\\

### 2. Log Completed Consultation (POST /log-consultation)
Call when a doctor finishes a consultation (Next Patient). Appends the record to quickappoint_demo_dataset.csv and automatically retrains & hot-swaps the model every 25 records.

**Request:** Same 7 fields + "actual_duration_mins": 18.5

### 3. Manual Retrain (POST /retrain)
Retrains the Random Forest model on quickappoint_demo_dataset.csv and hot-swaps quickappoint_rf_model.joblib in memory with zero downtime.
