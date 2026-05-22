"""
Analytics Routes: /api/analytics/*
User statistics, login history, ML performance metrics
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sqlite3
import json
import datetime

from database import get_db
from utils.auth_utils import get_current_user_id

router = APIRouter()
security = HTTPBearer()


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


@router.get("/dashboard")
async def get_dashboard(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get user analytics dashboard data"""
    user_id = current_user['id']
    
    # Login attempts
    attempts = db.execute(
        """SELECT password_correct, ml_score, ml_decision, risk_level, confidence, created_at
           FROM login_attempts WHERE user_id = ? ORDER BY created_at DESC LIMIT 30""",
        (user_id,)
    ).fetchall()
    
    attempts_list = [dict(a) for a in attempts]
    
    # Success rate
    if attempts_list:
        successful = sum(1 for a in attempts_list if a['password_correct'])
        success_rate = round(successful / len(attempts_list) * 100, 1)
        
        # Genuine predictions
        ml_attempts = [a for a in attempts_list if a['ml_score'] is not None]
        genuine_rate = 0
        if ml_attempts:
            genuine = sum(1 for a in ml_attempts if a['ml_decision'] == 'GENUINE')
            genuine_rate = round(genuine / len(ml_attempts) * 100, 1)
    else:
        success_rate = 100
        genuine_rate = 100
    
    # Typing samples count
    training_count = db.execute(
        "SELECT COUNT(*) as cnt FROM typing_samples WHERE user_id = ? AND session_type = 'training'",
        (user_id,)
    ).fetchone()['cnt']
    
    # ML model info
    model_info = db.execute(
        """SELECT * FROM ml_models WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
        (user_id,)
    ).fetchone()
    
    model_data = None
    if model_info:
        model_data = {
            "accuracy": round(model_info['accuracy'] * 100, 1) if model_info['accuracy'] else None,
            "precision": round(model_info['precision_score'] * 100, 1) if model_info['precision_score'] else None,
            "recall": round(model_info['recall_score'] * 100, 1) if model_info['recall_score'] else None,
            "f1": round(model_info['f1_score'] * 100, 1) if model_info['f1_score'] else None,
            "training_samples": model_info['training_samples'],
            "created_at": model_info['created_at']
        }
    
    # Risk distribution
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for a in attempts_list:
        if a['risk_level'] in risk_dist:
            risk_dist[a['risk_level']] += 1
    
    return {
        "user": {
            "username": current_user['username'],
            "training_complete": bool(current_user['training_complete']),
            "training_samples": training_count
        },
        "stats": {
            "total_logins": len(attempts_list),
            "success_rate": success_rate,
            "genuine_rate": genuine_rate,
            "risk_distribution": risk_dist
        },
        "recent_attempts": attempts_list[:10],
        "model": model_data
    }


@router.get("/history")
async def get_login_history(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get detailed login history"""
    attempts = db.execute(
        """SELECT * FROM login_attempts WHERE user_id = ? 
           ORDER BY created_at DESC LIMIT 100""",
        (current_user['id'],)
    ).fetchall()
    
    return {
        "history": [dict(a) for a in attempts],
        "total": len(attempts)
    }


@router.get("/typing-profile")
async def get_typing_profile(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get user's typing behavioral profile"""
    samples = db.execute(
        """SELECT features, created_at FROM typing_samples 
           WHERE user_id = ? AND session_type = 'training'
           ORDER BY created_at DESC LIMIT 20""",
        (current_user['id'],)
    ).fetchall()
    
    if not samples:
        return {"profile": None, "message": "No training data available"}
    
    import numpy as np
    
    all_features = [json.loads(s['features']) for s in samples]
    
    # Aggregate profile statistics
    feature_keys = ['hold_mean', 'hold_std', 'flight_mean', 'flight_std', 
                    'typing_speed', 'rhythm_consistency']
    
    profile = {}
    for key in feature_keys:
        values = [f.get(key, 0) for f in all_features if f.get(key) is not None]
        if values:
            profile[key] = {
                "mean": round(float(np.mean(values)), 3),
                "std": round(float(np.std(values)), 3),
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3)
            }
    
    # Timeline data for charts
    timeline = [
        {
            "timestamp": s['created_at'],
            "typing_speed": json.loads(s['features']).get('typing_speed', 0),
            "rhythm": json.loads(s['features']).get('rhythm_consistency', 0)
        }
        for s in samples
    ]
    
    return {
        "profile": profile,
        "timeline": timeline,
        "sample_count": len(samples)
    }
