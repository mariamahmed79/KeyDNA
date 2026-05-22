"""
Keystroke Dynamics Routes: /api/keystroke/*
Training, prediction, and biometric management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import json
import datetime

from database import get_db
from utils.auth_utils import get_current_user_id
from ml.feature_extractor import extract_features, features_to_vector
from ml.authenticator import KeystrokeAuthenticator

router = APIRouter()
security = HTTPBearer()

REQUIRED_TRAINING_SAMPLES = 8


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db)
):
    user_id = get_current_user_id(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)


class KeystrokeSampleRequest(BaseModel):
    events: List[Dict[str, Any]]
    session_type: str = "training"  # "training" or "predict"


class TrainModelRequest(BaseModel):
    force_retrain: bool = False


@router.post("/sample")
async def submit_sample(
    req: KeystrokeSampleRequest,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Submit a keystroke sample for training or prediction"""
    if not req.events or len(req.events) < 4:
        raise HTTPException(status_code=400, detail="Insufficient keystroke data")
    
    # Extract features
    features = extract_features(req.events)
    if not features:
        raise HTTPException(status_code=400, detail="Failed to extract features")
    
    feature_vector = features_to_vector(features)
    
    # Store sample
    db.execute(
        """INSERT INTO typing_samples (user_id, session_type, features, raw_events, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            current_user['id'],
            req.session_type,
            json.dumps(features),
            json.dumps(req.events[:50]),  # Store only first 50 events
            datetime.datetime.utcnow()
        )
    )
    
    # Count training samples
    sample_count = db.execute(
        "SELECT COUNT(*) as cnt FROM typing_samples WHERE user_id = ? AND session_type = 'training'",
        (current_user['id'],)
    ).fetchone()['cnt']
    
    db.execute(
        "UPDATE users SET training_samples = ? WHERE id = ?",
        (sample_count, current_user['id'])
    )
    db.commit()
    
    progress = min(100, int((sample_count / REQUIRED_TRAINING_SAMPLES) * 100))
    
    return {
        "success": True,
        "features": features,
        "sample_count": sample_count,
        "required_samples": REQUIRED_TRAINING_SAMPLES,
        "progress": progress,
        "ready_to_train": sample_count >= REQUIRED_TRAINING_SAMPLES
    }


@router.post("/train")
async def train_model(
    req: TrainModelRequest = TrainModelRequest(),
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Train ML model on user's keystroke samples"""
    # Get all training samples
    samples = db.execute(
        """SELECT features FROM typing_samples 
           WHERE user_id = ? AND session_type = 'training'
           ORDER BY created_at DESC""",
        (current_user['id'],)
    ).fetchall()
    
    if len(samples) < REQUIRED_TRAINING_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Need {REQUIRED_TRAINING_SAMPLES} samples, have {len(samples)}"
        )
    
    # Build feature matrix
    feature_vectors = []
    for sample in samples:
        features = json.loads(sample['features'])
        vector = features_to_vector(features)
        feature_vectors.append(vector)
    
    # Train model
    auth = KeystrokeAuthenticator(current_user['id'])
    result = auth.train(feature_vectors)
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', 'Training failed'))
    
    # Update user training status
    metrics = result.get('metrics', {})
    db.execute(
        "UPDATE users SET training_complete = TRUE WHERE id = ?",
        (current_user['id'],)
    )
    
    # Save model metadata
    db.execute(
        """INSERT INTO ml_models 
           (user_id, model_type, accuracy, precision_score, recall_score, f1_score, training_samples, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            current_user['id'],
            'ensemble',
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            len(samples),
            datetime.datetime.utcnow()
        )
    )
    db.commit()
    
    return {
        "success": True,
        "message": "Model trained successfully",
        "metrics": metrics,
        "training_samples": len(samples)
    }


@router.post("/predict")
async def predict(
    req: KeystrokeSampleRequest,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Predict authenticity of keystroke sample"""
    if not current_user['training_complete']:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Extract features
    features = extract_features(req.events)
    if not features:
        raise HTTPException(status_code=400, detail="Failed to extract features")
    
    feature_vector = features_to_vector(features)
    
    # Run prediction
    auth = KeystrokeAuthenticator(current_user['id'])
    result = auth.predict(feature_vector)
    
    # Store prediction sample
    db.execute(
        """INSERT INTO typing_samples (user_id, session_type, features, raw_events, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            current_user['id'], 'predict',
            json.dumps(features),
            json.dumps(req.events[:50]),
            datetime.datetime.utcnow()
        )
    )
    db.commit()
    
    return {
        "success": True,
        "features": features,
        "prediction": result
    }


@router.get("/samples")
async def get_samples(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get user's keystroke samples history"""
    samples = db.execute(
        """SELECT id, session_type, features, created_at 
           FROM typing_samples 
           WHERE user_id = ?
           ORDER BY created_at DESC 
           LIMIT 50""",
        (current_user['id'],)
    ).fetchall()
    
    return {
        "samples": [
            {
                "id": s['id'],
                "session_type": s['session_type'],
                "features": json.loads(s['features']),
                "created_at": s['created_at']
            }
            for s in samples
        ],
        "total": len(samples)
    }


@router.delete("/reset")
async def reset_training(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Reset user's training data and model"""
    import shutil
    from pathlib import Path
    
    # Delete samples
    db.execute("DELETE FROM typing_samples WHERE user_id = ?", (current_user['id'],))
    db.execute("DELETE FROM ml_models WHERE user_id = ?", (current_user['id'],))
    db.execute(
        "UPDATE users SET training_complete = FALSE, training_samples = 0 WHERE id = ?",
        (current_user['id'],)
    )
    db.commit()
    
    # Delete model files
    model_path = Path(f"ml_models/user_{current_user['id']}")
    if model_path.exists():
        shutil.rmtree(model_path)
    
    return {"success": True, "message": "Training data reset successfully"}
