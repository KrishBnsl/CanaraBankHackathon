# api.py
import os
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from collections import Counter

# --- Import your custom classes and utilities ---
from model_definition import MultimodalEmbeddingModel
from main_db import MongoAuthenticator # Import the updated MongoDB class

# =========================================================================================
# --- 1. API Setup & Configuration ---
# =========================================================================================

app = FastAPI(
    title="Multimodal Biometric Authentication API",
    description="API using Adaptive Thresholding Heuristic (ATH) for verification.",
    version="1.2.0"
)

# --- Model & Device Config ---
DEVICE = torch.device("cpu")
MAX_LEN = 1000
MODEL_PARAMS = {
    'hidden_dim': 128, 'proj_dim': 128, 'tcn_layers': 5,
    'dropout_rate': 0.4, 'sequence_length': MAX_LEN
}
SENSOR_LIST = [
    'key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro',
    'sensor_lacc', 'sensor_magn', 'sensor_nacc'
]
SENSOR_DIMS = { 'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3, 'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3 }
MAX_FEATURE_DIM = max(SENSOR_DIMS.values())
# Sensor used to build the statistical profile for ATH
PROFILE_SENSOR = 'sensor_lacc'

# --- Adaptive Thresholding Heuristic (ATH) Config ---
# As per the paper, these are tunable.
ATH_PERIODICITY_LIMIT = 4 # Permitted periodic outlier occurrences.
ATH_PROPORTION_LIMIT = 0.02 # Permitted proportion of data to be anomalous (1%).

# =========================================================================================
# --- 2. Load Model & Connect to Database ---
# =========================================================================================

MODEL_FILE = "multimodal_authentication_model.pkl"
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"[ERROR] Model file '{MODEL_FILE}' not found.")

model = MultimodalEmbeddingModel(SENSOR_LIST, SENSOR_DIMS, MODEL_PARAMS)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print("[INFO] Model loaded and ready for inference.")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load model: {e}")

db_authenticator = MongoAuthenticator()

# =========================================================================================
# --- 3. Helper Functions & Schemas ---
# =========================================================================================

class SensorData(BaseModel):
    data: Dict[str, List[List[float]]] = Field(default_factory=dict)

class UserRequest(BaseModel):
    user_id: str
    sensor_data: SensorData

class AuthResponse(BaseModel):
    authenticated: bool
    max_anomaly_score: float # Lower is better. Represents max Z-score.
    dynamic_threshold: float
    message: str

def preprocess_input(sensor_data: SensorData) -> torch.Tensor:
    tensors = []
    for sensor in SENSOR_LIST:
        readings = sensor_data.data.get(sensor, [])
        if not readings: data = torch.zeros(MAX_LEN, SENSOR_DIMS[sensor])
        else:
            df = pd.DataFrame(readings)
            if not df.empty and df.std().sum() > 0: df = (df - df.mean()) / df.std().replace(0, 1)
            df.fillna(0, inplace=True)
            data = torch.tensor(df.values, dtype=torch.float32)
        T, D = data.shape
        if T > MAX_LEN: data = data[:MAX_LEN]
        elif T < MAX_LEN: data = torch.cat([data, torch.zeros(MAX_LEN - T, D)], dim=0)
        tensors.append(data)
    padded = []
    for t in tensors:
        if t.shape[1] < MAX_FEATURE_DIM:
            pad = torch.zeros(t.shape[0], MAX_FEATURE_DIM - t.shape[1])
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.stack(padded).unsqueeze(0)

def apply_ath(scores: np.ndarray) -> float:
    """
    Applies the Adaptive Thresholding Heuristic to find a dynamic threshold.
    Based on Algorithm 1 from the source paper.
    """
    if len(scores) == 0: return float('inf')
    
    # Sort unique scores in descending order to test thresholds
    threshold_candidates = sorted(np.unique(scores), reverse=True)
    
    final_threshold = threshold_candidates[0] if threshold_candidates else float('inf')
    
    for thresh in threshold_candidates:
        # Get indices of outliers for the current threshold
        outlier_indices = np.where(scores >= thresh)[0]
        
        # Condition 1: Check proportion limit
        proportion = len(outlier_indices) / len(scores)
        if proportion > ATH_PROPORTION_LIMIT:
            break # Proportionality violated, stop and use previous threshold

        # Condition 2: Check periodicity limit
        if len(outlier_indices) > 1:
            # Get temporal differences between consecutive outliers
            diffs = np.diff(outlier_indices)
            diff_counts = Counter(diffs)
            
            # If any difference occurs too frequently, it's periodic
            if diff_counts and max(diff_counts.values()) > ATH_PERIODICITY_LIMIT:
                break # Periodicity violated, stop

        # If both conditions are met, this threshold is valid. Save it and continue.
        final_threshold = thresh
        
    return final_threshold

# =========================================================================================
# --- 4. API Endpoints ---
# =========================================================================================

@app.post("/enroll")
async def enroll(request: UserRequest):
    """
    Generates an embedding and a statistical profile, saving them to the database.
    """
    try:
        # 1. Generate embedding (as before)
        input_tensor = preprocess_input(request.sensor_data).to(DEVICE)
        with torch.no_grad():
            embedding = model(input_tensor).cpu().numpy().flatten().tolist()

        # 2. Create statistical profile for ATH
        profile_data = request.sensor_data.data.get(PROFILE_SENSOR)
        if not profile_data:
            raise HTTPException(status_code=400, detail=f"Profile sensor '{PROFILE_SENSOR}' not found in data.")
        
        # Calculate magnitude and then mean/std
        magnitudes = np.linalg.norm(np.array(profile_data), axis=1)
        profile_mean = np.mean(magnitudes)
        profile_std = np.std(magnitudes)
        
        # 3. Save the complete profile to MongoDB
        success = db_authenticator.enroll_user(
            request.user_id, embedding, profile_mean, profile_std
        )
        if success:
            return {"status": "success", "message": f"User '{request.user_id}' enrolled successfully."}
        else:
            raise HTTPException(status_code=500, detail="Database operation failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment error: {str(e)}")


@app.post("/verify", response_model=AuthResponse)
async def verify(request: UserRequest):
    """
    Verifies a user by generating Z-scores and using ATH to set a dynamic threshold.
    """
    try:
        # 1. Get the user's stored statistical profile from the database
        user_profile = db_authenticator.get_user_profile(request.user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found. Please enroll first.")
        
        stored_mean = user_profile['profile_mean']
        stored_std = user_profile.get('profile_std', 1.0) # Use 1.0 if std is missing or zero
        if stored_std == 0: stored_std = 1.0

        # 2. Process live sensor data to get Z-scores
        live_data = request.sensor_data.data.get(PROFILE_SENSOR)
        if not live_data:
            raise HTTPException(status_code=400, detail=f"Profile sensor '{PROFILE_SENSOR}' not found in data.")
            
        live_magnitudes = np.linalg.norm(np.array(live_data), axis=1)
        z_scores = np.abs((live_magnitudes - stored_mean) / stored_std) # Use absolute Z-scores

        # 3. Use ATH to get a dynamic anomaly threshold for the Z-scores
        dynamic_threshold = apply_ath(z_scores)

        # 4. Check if any live Z-score exceeds the dynamic threshold
        max_score = np.max(z_scores) if len(z_scores) > 0 else 0.0
        is_authenticated = max_score < dynamic_threshold
        
        message = "User authenticated successfully." if is_authenticated else "Authentication failed. Anomalous behavior detected."
        
        return {
            "authenticated": is_authenticated,
            "max_anomaly_score": max_score,
            "dynamic_threshold": dynamic_threshold,
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Multimodal Authentication API. Visit /docs for usage."}
