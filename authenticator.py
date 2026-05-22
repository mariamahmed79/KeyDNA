"""
KeyDNA Machine Learning Authentication Engine
Trains user-specific models using KNN, Random Forest, and SVM
Implements one-class and binary classification for impostor detection
"""

import numpy as np
import json
import os
import joblib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = Path("ml_models")
MODELS_DIR.mkdir(exist_ok=True)

MIN_TRAINING_SAMPLES = 5


class KeystrokeAuthenticator:
    """
    Multi-model keystroke dynamics authenticator.
    Uses ensemble of KNN, RandomForest, and OneClass-SVM for robust detection.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.scaler = StandardScaler()
        self.knn = None
        self.rf = None
        self.ocsvm = None
        self.isolation_forest = None
        self.is_trained = False
        self.training_stats = {}
        self.model_path = MODELS_DIR / f"user_{user_id}"
    
    def train(self, feature_vectors: List[List[float]]) -> Dict[str, Any]:
        """
        Train authentication models on user's typing samples.
        Returns training metrics and model performance.
        """
        if len(feature_vectors) < MIN_TRAINING_SAMPLES:
            return {
                "success": False,
                "error": f"Need at least {MIN_TRAINING_SAMPLES} samples, got {len(feature_vectors)}"
            }
        
        X = np.array(feature_vectors)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit scaler on genuine user data
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate synthetic impostor samples for binary classification
        X_impostor = self._generate_impostor_samples(X_scaled, n_samples=len(X) * 3)
        
        # Combined dataset for binary classifiers
        X_combined = np.vstack([X_scaled, X_impostor])
        y_combined = np.array([1] * len(X_scaled) + [0] * len(X_impostor))
        
        # Train KNN
        k = min(5, len(X_scaled))
        self.knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
        self.knn.fit(X_combined, y_combined)
        
        # Train Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        self.rf.fit(X_combined, y_combined)
        
        # Train One-Class SVM (anomaly detection on genuine data only)
        nu = max(0.05, 1.0 / len(X_scaled))
        self.ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        self.ocsvm.fit(X_scaled)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(X_scaled)
        
        # Evaluate models
        metrics = self._evaluate_models(X_scaled, X_impostor)
        
        self.is_trained = True
        self.training_stats = {
            "n_samples": len(feature_vectors),
            "n_features": X.shape[1],
            **metrics
        }
        
        # Save models
        self._save_models()
        
        return {
            "success": True,
            "metrics": self.training_stats
        }
    
    def predict(self, feature_vector: List[float]) -> Dict[str, Any]:
        """
        Predict whether a typing sample belongs to the genuine user.
        Returns confidence score, risk level, and decision.
        """
        if not self.is_trained and not self._load_models():
            return {
                "authenticated": False,
                "confidence": 0.0,
                "risk_level": "HIGH",
                "message": "Model not trained yet",
                "scores": {}
            }
        
        x = np.array(feature_vector).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self.scaler.transform(x)
        
        scores = {}
        
        # KNN prediction
        if self.knn:
            knn_prob = self.knn.predict_proba(x_scaled)[0]
            scores['knn'] = float(knn_prob[1]) if len(knn_prob) > 1 else 0.5
        
        # Random Forest prediction
        if self.rf:
            rf_prob = self.rf.predict_proba(x_scaled)[0]
            scores['random_forest'] = float(rf_prob[1]) if len(rf_prob) > 1 else 0.5
        
        # One-Class SVM (1=genuine, -1=anomaly)
        if self.ocsvm:
            ocsvm_pred = self.ocsvm.predict(x_scaled)[0]
            ocsvm_score = self.ocsvm.decision_function(x_scaled)[0]
            scores['ocsvm'] = float(1.0 / (1.0 + np.exp(-ocsvm_score)))  # sigmoid
        
        # Isolation Forest
        if self.isolation_forest:
            if_score = self.isolation_forest.score_samples(x_scaled)[0]
            scores['isolation_forest'] = float(1.0 / (1.0 + np.exp(-if_score * 5)))
        
        # Weighted ensemble
        weights = {'knn': 0.30, 'random_forest': 0.35, 'ocsvm': 0.20, 'isolation_forest': 0.15}
        confidence = sum(scores.get(k, 0.5) * w for k, w in weights.items())
        
        # Decision with threshold
        THRESHOLD = 0.55
        authenticated = confidence >= THRESHOLD
        
        # Risk level
        if confidence >= 0.85:
            risk_level = "LOW"
        elif confidence >= 0.65:
            risk_level = "MEDIUM"
        elif confidence >= 0.45:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            "authenticated": authenticated,
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "scores": {k: round(v * 100, 2) for k, v in scores.items()},
            "threshold": THRESHOLD * 100,
            "decision": "GENUINE" if authenticated else "IMPOSTOR"
        }
    
    def _generate_impostor_samples(self, X_genuine: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate synthetic impostor samples by perturbing genuine data"""
        n_features = X_genuine.shape[1]
        impostors = []
        
        for _ in range(n_samples):
            # Random perturbation with large offset
            base = X_genuine[np.random.randint(len(X_genuine))]
            noise_scale = np.random.uniform(1.5, 4.0)
            noise = np.random.normal(0, noise_scale, n_features)
            sample = base + noise
            impostors.append(sample)
        
        # Also add completely random samples
        random_samples = np.random.normal(
            loc=np.random.uniform(-3, 3, n_features),
            scale=np.random.uniform(0.5, 2.0, n_features),
            size=(n_samples // 2, n_features)
        )
        
        return np.vstack([np.array(impostors), random_samples])
    
    def _evaluate_models(self, X_genuine: np.ndarray, X_impostor: np.ndarray) -> Dict:
        """Evaluate model performance"""
        # Use a subset of impostors equal to genuine samples
        n = len(X_genuine)
        X_imp_subset = X_impostor[:n]
        
        X_eval = np.vstack([X_genuine, X_imp_subset])
        y_eval = np.array([1] * n + [0] * n)
        
        metrics = {}
        
        if self.rf:
            y_pred = self.rf.predict(X_eval)
            metrics['accuracy'] = round(float(accuracy_score(y_eval, y_pred)), 4)
            metrics['precision'] = round(float(precision_score(y_eval, y_pred, zero_division=0)), 4)
            metrics['recall'] = round(float(recall_score(y_eval, y_pred, zero_division=0)), 4)
            metrics['f1'] = round(float(f1_score(y_eval, y_pred, zero_division=0)), 4)
        
        return metrics
    
    def _save_models(self):
        """Persist trained models to disk"""
        self.model_path.mkdir(exist_ok=True)
        joblib.dump(self.scaler, self.model_path / "scaler.pkl")
        if self.knn: joblib.dump(self.knn, self.model_path / "knn.pkl")
        if self.rf: joblib.dump(self.rf, self.model_path / "rf.pkl")
        if self.ocsvm: joblib.dump(self.ocsvm, self.model_path / "ocsvm.pkl")
        if self.isolation_forest: joblib.dump(self.isolation_forest, self.model_path / "isoforest.pkl")
    
    def _load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            if not self.model_path.exists():
                return False
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            knn_path = self.model_path / "knn.pkl"
            rf_path = self.model_path / "rf.pkl"
            ocsvm_path = self.model_path / "ocsvm.pkl"
            if_path = self.model_path / "isoforest.pkl"
            if knn_path.exists(): self.knn = joblib.load(knn_path)
            if rf_path.exists(): self.rf = joblib.load(rf_path)
            if ocsvm_path.exists(): self.ocsvm = joblib.load(ocsvm_path)
            if if_path.exists(): self.isolation_forest = joblib.load(if_path)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
