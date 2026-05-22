"""
Authentication Routes: /api/auth/*
Registration, Login, Profile management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, EmailStr
from typing import Optional, List
import sqlite3
import json
import datetime

from database import get_db
from utils.auth_utils import (
    hash_password, verify_password,
    create_access_token, get_current_user_id
)

router = APIRouter()
security = HTTPBearer()


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    
    @validator('username')
    def username_valid(cls, v):
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be 3-30 characters")
        if not v.isalnum() and '_' not in v:
            raise ValueError("Username must be alphanumeric")
        return v.lower()
    
    @validator('password')
    def password_valid(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str
    keystroke_data: Optional[List[dict]] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def get_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db)
):
    user_id = get_current_user_id(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return dict(user)


@router.post("/register", status_code=201)
async def register(req: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    """Register a new user account"""
    # Check existing username
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (req.username,)
    ).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    
    # Check existing email
    existing_email = db.execute(
        "SELECT id FROM users WHERE email = ?", (req.email,)
    ).fetchone()
    if existing_email:
        raise HTTPException(status_code=409, detail="Email already registered")
    
    # Hash password and create user
    password_hash = hash_password(req.password)
    
    cursor = db.execute(
        """INSERT INTO users (username, email, password_hash, created_at)
           VALUES (?, ?, ?, ?)""",
        (req.username, req.email, password_hash, datetime.datetime.utcnow())
    )
    db.commit()
    user_id = cursor.lastrowid
    
    # Create token
    token = create_access_token({"user_id": user_id, "username": req.username})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "username": req.username,
            "email": req.email,
            "training_complete": False,
            "training_samples": 0
        },
        "message": "Account created successfully"
    }


@router.post("/login")
async def login(req: LoginRequest, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Login with username, password, and optional keystroke data"""
    # Find user
    user = db.execute(
        "SELECT * FROM users WHERE username = ?", (req.username.lower(),)
    ).fetchone()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = dict(user)
    
    # Verify password
    password_correct = verify_password(req.password, user['password_hash'])
    
    if not password_correct:
        # Log failed attempt
        db.execute(
            """INSERT INTO login_attempts 
               (user_id, username, password_correct, ml_score, risk_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user['id'], req.username, False, None, "CRITICAL", datetime.datetime.utcnow())
        )
        db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if keystroke analysis needed
    ml_result = None
    if req.keystroke_data and user['training_complete']:
        from ml.feature_extractor import extract_features, features_to_vector
        from ml.authenticator import KeystrokeAuthenticator
        
        features = extract_features(req.keystroke_data)
        if features:
            feature_vector = features_to_vector(features)
            auth = KeystrokeAuthenticator(user['id'])
            ml_result = auth.predict(feature_vector)
    
    # Log attempt
    db.execute(
        """INSERT INTO login_attempts 
           (user_id, username, password_correct, ml_score, ml_decision, risk_level, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user['id'], req.username, True,
            ml_result['confidence'] if ml_result else None,
            ml_result['decision'] if ml_result else "PASSWORD_ONLY",
            ml_result['risk_level'] if ml_result else "LOW",
            ml_result['confidence'] if ml_result else None,
            datetime.datetime.utcnow()
        )
    )
    
    # Update last login
    db.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.datetime.utcnow(), user['id'])
    )
    db.commit()
    
    token = create_access_token({"user_id": user['id'], "username": user['username']})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user['id'],
            "username": user['username'],
            "email": user['email'],
            "training_complete": bool(user['training_complete']),
            "training_samples": user['training_samples']
        },
        "ml_analysis": ml_result,
        "message": "Login successful"
    }


@router.get("/me")
async def get_profile(current_user: dict = Depends(get_user_from_token)):
    """Get current user profile"""
    return {
        "id": current_user['id'],
        "username": current_user['username'],
        "email": current_user['email'],
        "training_complete": bool(current_user['training_complete']),
        "training_samples": current_user['training_samples'],
        "created_at": current_user['created_at'],
        "last_login": current_user['last_login']
    }
