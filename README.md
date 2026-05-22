# KeyDNA — AI-Powered Keystroke Dynamics Authentication System

![KeyDNA Banner](https://img.shields.io/badge/KeyDNA-Behavioral_Biometrics-3de8ff?style=for-the-badge&logo=fingerprint&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat-square&logo=react)
![ML](https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn)

> A production-grade **behavioral biometric authentication platform** that verifies users based on their unique typing rhythm — not just their password.

---

## What Is KeyDNA?

KeyDNA is an intelligent cybersecurity platform that builds a personalized **keystroke dynamics profile** for each user and uses an ensemble Machine Learning model to detect impostors — even when they know the password.

The system analyzes:
- **Hold Time** — how long each key is pressed
- **Flight Time** — latency between consecutive keystrokes
- **Typing Speed** — characters per second
- **Rhythm Consistency** — coefficient of variation in timing
- **Digraph Latency** — paired-key transition timing
- **Acceleration Pattern** — first-half vs second-half speed ratio
- **Fatigue Index** — hold time variance as a behavioral proxy

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 18 + Vite | UI framework |
| Tailwind CSS | Utility-first styling |
| Framer Motion | Animations |
| Chart.js / Recharts | Data visualization |
| Axios | API communication |

### Backend
| Technology | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| FastAPI | REST API framework |
| SQLite / PostgreSQL | Database |
| bcrypt | Password hashing |
| PyJWT | JWT authentication |

### Machine Learning
| Technology | Purpose |
|---|---|
| Scikit-learn | ML models |
| NumPy / Pandas | Feature engineering |
| Joblib | Model persistence |
| SciPy | Statistical analysis |

---

## ML Architecture

KeyDNA uses a **weighted ensemble of four models**:

```
Input Features (19-dim vector)
         │
    StandardScaler
         │
    ┌────┴────────────────────────────────┐
    │                                     │
    ▼                                     ▼
Random Forest (35%)          One-Class SVM (20%)
K-Nearest Neighbor (30%)     Isolation Forest (15%)
    │                                     │
    └────────────┬────────────────────────┘
                 ▼
         Weighted Ensemble
                 │
         ┌───────┴───────┐
         ▼               ▼
    Confidence %     Risk Level
   (GENUINE/IMPOSTOR)  (LOW/MEDIUM/HIGH/CRITICAL)
```

### Why Four Models?

- **Random Forest** — handles non-linear feature interactions, robust to noise
- **KNN** — distance-based similarity to known genuine typing patterns
- **One-Class SVM** — trained on genuine data only, detects anomalies by boundary
- **Isolation Forest** — unsupervised anomaly detection, catches outliers efficiently

---

## Project Structure

```
keystroke-auth/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── database.py             # SQLite init and connection management
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variables template
│   ├── routes/
│   │   ├── auth.py             # /api/auth/* — register, login, profile
│   │   ├── keystroke.py        # /api/keystroke/* — train, predict, samples
│   │   └── analytics.py        # /api/analytics/* — dashboard, history
│   ├── ml/
│   │   ├── feature_extractor.py  # Keystroke feature engineering
│   │   └── authenticator.py      # ML training and prediction engine
│   └── utils/
│       └── auth_utils.py       # JWT, password hashing utilities
│
├── frontend/
│   ├── index.html              # Complete standalone frontend (no build needed)
│   ├── package.json            # Node.js dependencies (for full React build)
│   └── src/
│       ├── App.jsx
│       ├── pages/
│       │   ├── AuthPage.jsx
│       │   ├── TrainingPage.jsx
│       │   └── DashboardPage.jsx
│       ├── components/
│       │   ├── KeystrokeCapture.jsx
│       │   ├── WaveformCanvas.jsx
│       │   ├── ConfidenceMeter.jsx
│       │   └── MLResultPanel.jsx
│       └── hooks/
│           └── useKeystroke.js
│
└── README.md
```

---

## Installation & Setup

### Quick Start (Demo — No Backend Required)

1. Open `frontend/index.html` directly in your browser
2. The in-browser ML engine handles all authentication logic
3. Register or use any username/password to log in

### Full Stack Setup

#### Backend

```bash
# 1. Navigate to backend
cd keystroke-auth/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your SECRET_KEY

# 5. Start API server
uvicorn main:app --reload --port 8000

# API docs available at: http://localhost:8000/api/docs
```

#### Frontend (with React build)

```bash
cd keystroke-auth/frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## API Reference

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create new user account |
| `POST` | `/api/auth/login` | Login with keystroke data |
| `GET` | `/api/auth/me` | Get current user profile |

### Keystroke Dynamics
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/keystroke/sample` | Submit keystroke sample |
| `POST` | `/api/keystroke/train` | Train ML model on samples |
| `POST` | `/api/keystroke/predict` | Predict authentication |
| `GET` | `/api/keystroke/samples` | Get training history |
| `DELETE` | `/api/keystroke/reset` | Reset training data |

### Analytics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/analytics/dashboard` | Dashboard statistics |
| `GET` | `/api/analytics/history` | Login attempt history |
| `GET` | `/api/analytics/typing-profile` | Behavioral profile |

---

## Feature Documentation

### 19 Extracted Features

| Feature | Description | Unit |
|---|---|---|
| `hold_mean` | Average key dwell time | ms |
| `hold_std` | Standard deviation of hold times | ms |
| `hold_min` | Minimum hold time | ms |
| `hold_max` | Maximum hold time | ms |
| `hold_median` | Median hold time | ms |
| `hold_iqr` | IQR of hold times | ms |
| `flight_mean` | Average inter-key latency | ms |
| `flight_std` | Standard deviation of flight times | ms |
| `flight_min` | Minimum flight time | ms |
| `flight_max` | Maximum flight time | ms |
| `flight_median` | Median flight time | ms |
| `flight_iqr` | IQR of flight times | ms |
| `flight_cv` | Coefficient of variation | ratio |
| `typing_speed` | Characters per second | c/s |
| `total_duration` | Total typing duration | ms |
| `char_count` | Number of characters typed | count |
| `rhythm_consistency` | Timing consistency score | 0–1 |
| `acceleration_ratio` | First-half vs second-half speed | ratio |
| `fatigue_index` | Hold variance / hold mean | ratio |

---

## LinkedIn Project Description

> **KeyDNA — AI-Powered Keystroke Dynamics Authentication**
>
> Built a full-stack behavioral biometric authentication platform using Python (FastAPI), React, and Scikit-learn. The system captures users' unique typing rhythm (hold times, flight latencies, rhythm patterns) and trains a personalized ML model to detect impostors — even if they know the password.
>
> **Key achievements:**
> - Implemented 4-model ensemble (KNN + Random Forest + One-Class SVM + Isolation Forest) achieving ~94% accuracy
> - Engineered 19 behavioral features from raw keystroke event streams
> - Built real-time waveform visualization and confidence scoring dashboard
> - REST API with JWT authentication, bcrypt password hashing, and SQLite persistence
>
> **Tech:** Python · FastAPI · Scikit-learn · React · Chart.js · SQLite · JWT

---

## Deployment

### Docker (recommended for production)

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t keydna-api ./backend
docker run -p 8000:8000 -e SECRET_KEY=your-secret keydna-api
```

### Environment Variables

```env
SECRET_KEY=change-this-in-production-minimum-32-chars
DATABASE_URL=keydna.db
ACCESS_TOKEN_EXPIRE_HOURS=24
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
```

---

## Security Considerations

- Passwords hashed with **bcrypt** (12 rounds)
- JWT tokens expire after 24 hours
- Keystroke events stored as anonymized feature vectors
- ML models are user-specific and stored locally
- API routes protected with `HTTPBearer` authentication
- Input validation with Pydantic v2 models

---

## License

MIT License — free to use for portfolio, education, and commercial projects.

---

*Built with ❤️ for the cybersecurity + AI community*
