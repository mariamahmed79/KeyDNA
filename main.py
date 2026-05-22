"""
KeyDNA - AI-Powered Keystroke Dynamics Authentication
Backend API - FastAPI + SQLite + Scikit-learn
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import sqlite3
import hashlib
import jwt
import datetime
import json
import numpy as np
from pathlib import Path
import logging

from routes.auth import router as auth_router
from routes.keystroke import router as keystroke_router
from routes.analytics import router as analytics_router
from database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KeyDNA API",
    description="AI-Powered Keystroke Dynamics Authentication System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("KeyDNA API started successfully")

@app.get("/")
async def root():
    return {
        "name": "KeyDNA API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(keystroke_router, prefix="/api/keystroke", tags=["Keystroke Dynamics"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
