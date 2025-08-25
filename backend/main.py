from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sqlite3
import hashlib
import jwt  # PyJWT
import datetime
import json
import io
import logging
from typing import Optional, Dict, Any, List
import os

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Secure ML Pipeline API", version="1.0.0")

# If you know your frontend origin(s), list them here instead of "*"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Catch-all handler to ensure any CORS preflight never falls into route logic.
@app.options("/{path:path}")
def cors_preflight_handler(path: str) -> Response:
    return Response(status_code=204)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Security / Auth
# -----------------------------------------------------------------------------
security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"

class LoginRequest(BaseModel):
    username: str
    password: str

class PipelineConfig(BaseModel):
    algorithm: Optional[str] = "isolation_forest"

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect('pipeline.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blockchain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            previous_hash TEXT,
            block_hash TEXT UNIQUE NOT NULL
        )
    ''')

    # Default user: admin / admin123
    password_hash = hashlib.sha256("admin123".encode()).hexdigest()
    cursor.execute(
        'INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)',
        ("admin", password_hash)
    )

    conn.commit()
    conn.close()

init_db()

# -----------------------------------------------------------------------------
# “Blockchain” logger
# -----------------------------------------------------------------------------
class SecurityLogger:
    @staticmethod
    def hash_data(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def get_previous_hash() -> str:
        conn = sqlite3.connect('pipeline.db')
        cursor = conn.cursor()
        cursor.execute('SELECT block_hash FROM blockchain_logs ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "genesis"

    @staticmethod
    def log_action(action: str, data: Dict[Any, Any]):
        data_str = json.dumps(data, sort_keys=True)
        data_hash = SecurityLogger.hash_data(data_str)
        previous_hash = SecurityLogger.get_previous_hash()

        timestamp = datetime.datetime.now().isoformat()
        block_content = f"{action}{data_hash}{previous_hash}{timestamp}"
        block_hash = SecurityLogger.hash_data(block_content)

        conn = sqlite3.connect('pipeline.db')
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO blockchain_logs (action, data_hash, timestamp, previous_hash, block_hash)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (action, data_hash, timestamp, previous_hash, block_hash)
        )
        conn.commit()
        conn.close()

        logger.info(f"Blockchain log created: {action} - {block_hash[:16]}...")

# -----------------------------------------------------------------------------
# ML Pipeline
# -----------------------------------------------------------------------------
class MLPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None

    def load_data(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        try:
            if filename.endswith('.csv'):
                self.data = pd.read_csv(io.BytesIO(file_content))
            elif filename.endswith('.json'):
                self.data = pd.read_json(io.BytesIO(file_content))
            else:
                raise ValueError("Unsupported file format. Use .csv or .json")

            SecurityLogger.log_action("DATA_UPLOAD", {
                "filename": filename,
                "rows": int(len(self.data)),
                "columns": int(len(self.data.columns))
            })

            return {
                "message": "Data loaded successfully",
                "rows": int(len(self.data)),
                "columns": int(len(self.data.columns)),
                "column_names": list(self.data.columns)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

    def run_anomaly_detection(self, algorithm: str = "isolation_forest") -> Dict[str, Any]:
        if self.data is None:
            raise HTTPException(status_code=400, detail="No data loaded")

        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("No numeric columns found for anomaly detection")

            numeric_data = numeric_data.fillna(numeric_data.mean())
            scaled_data = self.scaler.fit_transform(numeric_data)

            if algorithm == "isolation_forest":
                self.model = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = self.model.fit_predict(scaled_data)
            else:
                self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                anomaly_labels = self.model.fit_predict(scaled_data)

            anomalies = (anomaly_labels == -1).astype(int)

            self.results = self.data.copy()
            self.results['anomaly'] = anomalies
            # NOTE: If you want real scores, use decision_function_ / negative_outlier_factor_
            self.results['anomaly_score'] = np.where(anomalies == 1, 0.8, 0.1)

            anomaly_count = int(anomalies.sum())
            total_count = int(len(anomalies))

            anomaly_percentage = (anomaly_count / max(total_count, 1)) * 100
            if anomaly_percentage > 15:
                risk_level = "HIGH"
            elif anomaly_percentage > 5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            SecurityLogger.log_action("PIPELINE_EXECUTION", {
                "algorithm": algorithm,
                "anomaly_count": anomaly_count,
                "total_count": total_count,
                "risk_level": risk_level
            })

            return {
                "message": "Anomaly detection completed",
                "algorithm": algorithm,
                "total_records": total_count,
                "anomalies_detected": anomaly_count,
                "anomaly_percentage": round(anomaly_percentage, 2),
                "risk_level": risk_level
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in anomaly detection: {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        if self.results is None:
            raise HTTPException(status_code=400, detail="No results available. Run pipeline first.")

        results_dict: List[Dict[str, Any]] = self.results.fillna("").to_dict('records')

        anomaly_counts = self.results['anomaly'].value_counts()
        chart_data = {
            "anomaly_distribution": {
                "labels": ["Normal", "Anomaly"],
                "data": [int(anomaly_counts.get(0, 0)), int(anomaly_counts.get(1, 0))]
            }
        }

        numeric_cols = self.results.select_dtypes(include=[np.number]).columns.tolist()
        # Remove helper columns from candidates
        for col in ['anomaly', 'anomaly_score']:
            if col in numeric_cols:
                numeric_cols.remove(col)

        if len(numeric_cols) >= 2:
            chart_data["scatter_plot"] = {
                "x": self.results[numeric_cols[0]].fillna(0).tolist(),
                "y": self.results[numeric_cols[1]].fillna(0).tolist(),
                "anomalies": self.results['anomaly'].tolist(),
                "x_label": numeric_cols[0],
                "y_label": numeric_cols[1]
            }

        return {
            "data": results_dict,
            "charts": chart_data,
            "summary": {
                "total_records": len(results_dict),
                "anomalies": int(self.results['anomaly'].sum()),
                "columns": list(self.results.columns)
            }
        }

ml_pipeline = MLPipeline()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/auth/login")
async def login(credentials: LoginRequest):
    username = credentials.username.strip()
    password = credentials.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect('pipeline.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT username FROM users WHERE username = ? AND password_hash = ?',
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    SecurityLogger.log_action("USER_LOGIN", {"username": username})
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: str = Depends(verify_token)
):
    if not file.filename.lower().endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")

    content = await file.read()
    result = ml_pipeline.load_data(content, file.filename)
    return result

@app.post("/pipeline/run")
async def run_pipeline(
    config: PipelineConfig,
    current_user: str = Depends(verify_token)
):
    result = ml_pipeline.run_anomaly_detection(config.algorithm or "isolation_forest")
    return result

@app.get("/results")
async def get_results(current_user: str = Depends(verify_token)):
    return ml_pipeline.get_results()

@app.get("/security/logs")
async def get_security_logs(current_user: str = Depends(verify_token)):
    conn = sqlite3.connect('pipeline.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT action, data_hash, timestamp, block_hash
        FROM blockchain_logs
        ORDER BY id DESC
        LIMIT 50
    ''')
    logs = cursor.fetchall()
    conn.close()

    return {
        "logs": [
            {
                "action": log[0],
                "data_hash": log[1][:16] + "...",
                "timestamp": log[2],
                "block_hash": log[3][:16] + "..."
            }
            for log in logs
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/")
async def root():
    return {"message": "Secure ML Pipeline API", "version": "1.0.0"}

# Optional: silence favicon/manifest 404s in dev
@app.get("/favicon.ico")
async def favicon_dev():
    return Response(status_code=204)

@app.get("/manifest.json")
async def manifest_dev():
    return Response(status_code=204)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
