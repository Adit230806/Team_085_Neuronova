# Secure ML Pipeline Backend

A FastAPI-based backend for secure machine learning pipeline with anomaly detection, JWT authentication, and blockchain-style logging.

## Features

- **Machine Learning**: Anomaly detection using Isolation Forest and Local Outlier Factor
- **Security**: JWT authentication, SHA-256 hashing, blockchain-style logging
- **Database**: SQLite for user management and audit logs
- **File Processing**: Support for CSV and JSON file uploads
- **CORS**: Configured for frontend integration

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API**
   - API: http://127.0.0.1:8000
   - Interactive Docs: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

## Default Credentials

- **Username**: admin
- **Password**: admin123

## API Endpoints

### Authentication
- `POST /auth/login` - Login and get JWT token

### Data Processing
- `POST /upload` - Upload CSV/JSON files
- `POST /pipeline/run` - Run anomaly detection pipeline
- `GET /results` - Get pipeline results with charts

### Security
- `GET /security/logs` - View blockchain-style audit logs

### Health
- `GET /health` - Health check
- `GET /` - API info

## Usage Example

1. **Login**
   ```bash
   curl -X POST "http://127.0.0.1:8000/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username": "admin", "password": "admin123"}'
   ```

2. **Upload Data**
   ```bash
   curl -X POST "http://127.0.0.1:8000/upload" \
        -H "Authorization: Bearer YOUR_TOKEN" \
        -F "file=@sample_data/demo_data.csv"
   ```

3. **Run Pipeline**
   ```bash
   curl -X POST "http://127.0.0.1:8000/pipeline/run" \
        -H "Authorization: Bearer YOUR_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"algorithm": "isolation_forest"}'
   ```

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `password_hash`: SHA-256 hashed password
- `created_at`: Account creation timestamp

### Blockchain Logs Table
- `id`: Primary key
- `action`: Action performed
- `data_hash`: SHA-256 hash of action data
- `timestamp`: Action timestamp
- `previous_hash`: Hash of previous block
- `block_hash`: Unique hash for this block

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: SHA-256 for password storage
- **Audit Logging**: Blockchain-style immutable logs
- **CORS Protection**: Configured for specific origins
- **Input Validation**: File type and data validation

## ML Algorithms

1. **Isolation Forest**: Efficient for large datasets
2. **Local Outlier Factor**: Good for local anomaly detection

Both algorithms return anomaly scores and risk levels (LOW/MEDIUM/HIGH).