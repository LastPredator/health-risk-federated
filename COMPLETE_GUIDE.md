# Complete Guide: Health Risk Federated Learning - Part 1 & Part 2

This comprehensive guide covers everything you need to set up, run, and use both Part 1 (Data & Model) and Part 2 (MLOps Pipeline) of the Health Risk Federated Learning project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Part 1: Data Ingestion & AI Model](#part-1-data-ingestion--ai-model)
4. [Part 2: MLOps Pipeline](#part-2-mlops-pipeline)
5. [Quick Start](#quick-start)
6. [Detailed Setup](#detailed-setup)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Deployment Options](#deployment-options)

---

## Project Overview

This project implements a federated learning system for health risk prediction using:
- **Wearable device data** (heart rate, steps, sleep, etc.)
- **Environmental data** (air quality, weather)
- **Federated Learning** (Flower framework)
- **MLOps Pipeline** (Docker, Kubernetes, CI/CD, Monitoring)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Part 1: Data & Model                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Data         │  │ Federated    │  │ Health Risk  │  │
│  │ Simulation   │→ │ Learning     │→ │ Model       │  │
│  │ (Wearables,  │  │ (Flower)     │  │ (Logistic   │  │
│  │  Air Quality)│  │              │  │ Regression) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Part 2: MLOps Pipeline                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Docker       │  │ Kubernetes   │  │ CI/CD        │  │
│  │ Containers   │  │ Deployment   │  │ (GitHub      │  │
│  │              │  │              │  │ Actions)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ MLflow       │  │ Inference    │  │ Prometheus   │  │
│  │ Tracking     │  │ API          │  │ Monitoring   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software

1. **Python 3.10+**
   - Download from: https://www.python.org/downloads/
   - Verify: `python --version`

2. **Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Verify: `docker --version`

3. **Git**
   - Usually pre-installed
   - Verify: `git --version`

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 10GB free
- **CPU**: Multi-core recommended

---

## Part 1: Data Ingestion & AI Model

### Overview

Part 1 includes:
- Data simulation (wearables, air quality, weather)
- Federated learning implementation (Flower)
- Health risk prediction model
- Data drift detection

### Directory Structure

```
part1-data-model/
├── data_simulation/
│   ├── wearables.py          # Wearable device data simulation
│   ├── air_quality.py         # Environmental sensor data
│   └── weather.py             # Weather data simulation
├── federated_learning/
│   ├── server.py              # Federated learning server
│   ├── client.py              # Federated learning client
│   ├── data_loader.py         # Data loading utilities
│   └── drift_detector.py     # Data drift detection
├── models/
│   └── health_risk_model.py   # Health risk prediction model
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
└── run_federated.py          # Main script to run federated learning
```

### Installation

```bash
# Navigate to project root
cd health-risk-federated

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Run Federated Learning Locally

**Terminal 1 - Start Server:**
```bash
cd part1-data-model
python run_federated.py server
```

**Terminal 2 - Start Client 1:**
```bash
cd part1-data-model
python run_federated.py client 01
```

**Terminal 3 - Start Client 2:**
```bash
cd part1-data-model
python run_federated.py client 02
```

#### 2. Test Data Simulation

```python
from data_simulation.wearables import WearableSimulator
from data_simulation.air_quality import EnvironmentalSimulator

# Generate wearable data
wear_sim = WearableSimulator(num_patients=100)
health_data = wear_sim.generate_daily_data("2024-01-15", "hospital_01")

# Generate environmental data
env_sim = EnvironmentalSimulator(num_sensors=10)
env_data = env_sim.generate_sensor_data("hospital_01")

print(health_data.head())
print(env_data.head())
```

#### 3. Test Model

```python
from models.health_risk_model import HealthRiskModel
import numpy as np

# Create and train model
model = HealthRiskModel()
X = np.random.rand(100, 11)  # 11 features
y = np.random.randint(0, 2, 100)  # Binary labels

model.fit(X, y)
predictions = model.predict_proba(X)
print(predictions)
```

#### 4. Test Drift Detection

```python
from federated_learning.drift_detector import DriftMonitor
import pandas as pd

# Create reference data
ref_data = pd.DataFrame({...})  # Your reference dataset

# Create monitor
monitor = DriftMonitor(ref_data)

# Check for drift
current_data = pd.DataFrame({...})  # Current data
drift_detected = monitor.check_drift(current_data, threshold=0.5)
```

### Key Features

- **Federated Learning**: Train models across distributed nodes without sharing raw data
- **Data Simulation**: Generate realistic health and environmental data
- **Drift Detection**: Monitor data distribution changes using Evidently
- **Model Tracking**: MLflow integration for experiment tracking

---

## Part 2: MLOps Pipeline

### Overview

Part 2 provides complete MLOps infrastructure:
- Docker containerization
- Kubernetes deployment
- CI/CD pipeline (GitHub Actions)
- Model serving (FastAPI)
- Monitoring (Prometheus)
- Automated retraining

### Directory Structure

```
part2-mlops/
├── docker/
│   ├── Dockerfile.server      # Server container
│   ├── Dockerfile.client      # Client container
│   ├── Dockerfile.inference   # Inference API container
│   ├── Dockerfile.mlflow      # MLflow container
│   └── prometheus/
│       └── prometheus.yml      # Prometheus configuration
├── k8s/
│   ├── namespace.yaml         # Kubernetes namespace
│   ├── configmap.yaml         # Configuration
│   ├── mlflow-deployment.yaml # MLflow deployment
│   ├── federated-*-deployment.yaml # Server/client deployments
│   ├── inference-deployment.yaml # Inference service
│   ├── prometheus-deployment.yaml # Monitoring
│   └── cronjob-retraining.yaml # Scheduled retraining
├── mlops/
│   ├── inference_server.py    # FastAPI inference service
│   └── retraining_pipeline.py # Automated retraining
├── scripts/
│   ├── build-images.sh        # Build Docker images
│   ├── deploy-k8s.sh         # Deploy to Kubernetes
│   ├── run-local.sh          # Run locally
│   └── trigger-retraining.sh # Trigger retraining
└── docker-compose.yml         # Local development setup
```

---

## Quick Start

### Step 1: Install Prerequisites

1. Install Docker Desktop
2. Install Python 3.10+
3. Install dependencies: `pip install -r requirements.txt`

### Step 2: Build Docker Images

**From project root:**

```powershell
# Windows PowerShell
cd part2-mlops/scripts
.\build-images.sh

# Or manually:
docker build -t health-risk-federated:latest -f part2-mlops/docker/Dockerfile.server .
docker build -t health-risk-federated-client:latest -f part2-mlops/docker/Dockerfile.client .
docker build -t health-risk-federated-inference:latest -f part2-mlops/docker/Dockerfile.inference .
docker build -t health-risk-federated-mlflow:latest -f part2-mlops/docker/Dockerfile.mlflow part2-mlops
```

### Step 3: Start Services

**From project root:**

```powershell
docker-compose -f part2-mlops/docker-compose.yml up -d
```

### Step 4: Train a Model

```powershell
# Train and register model with MLflow
python part2-mlops/train_model_simple.py

# Restart inference service to load model
docker-compose -f part2-mlops/docker-compose.yml restart inference
```

### Step 5: Test the API

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8080/health

# Make prediction
$body = @{
    heart_rate = 75
    steps = 8000
    sleep_hours = 7.5
    respiratory_rate = 16
    body_temp = 98.6
    pm25 = 15
    pm10 = 25
    o3 = 0.05
    no2 = 0.02
    temperature = 72
    humidity = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
```

### Step 6: Access Services

- **MLflow UI**: http://localhost:5000
- **Inference API**: http://localhost:8080
- **Prometheus**: http://localhost:9090

---

## Detailed Setup

### Part 1 Setup

#### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `flwr==1.5.0` - Federated learning framework
- `mlflow==2.7.1` - Experiment tracking
- `evidently==0.4.0` - Data drift detection
- `scikit-learn==1.3.0` - Machine learning
- `pandas==2.0.3` - Data manipulation
- `torch==2.0.1` - Data loaders

#### 2. Run Federated Learning

**Option A: Local Execution (3 terminals)**

Terminal 1:
```bash
cd part1-data-model
python run_federated.py server
```

Terminal 2:
```bash
cd part1-data-model
python run_federated.py client 01
```

Terminal 3:
```bash
cd part1-data-model
python run_federated.py client 02
```

**Option B: Using Docker Compose**

```bash
cd part2-mlops
docker-compose up server client01 client02
```

#### 3. View Results in MLflow

1. Start MLflow: `mlflow ui` (or use Docker Compose)
2. Open http://localhost:5000
3. Navigate to "federated_health_risk" experiment
4. View metrics, parameters, and model artifacts

### Part 2 Setup

#### 1. Build All Docker Images

**Using Script (Recommended):**

```bash
# Unix/macOS/Linux
cd part2-mlops/scripts
chmod +x build-images.sh
./build-images.sh

# Windows (Git Bash or WSL)
cd part2-mlops/scripts
bash build-images.sh
```

**Manual Build:**

```bash
# From project root
docker build -t health-risk-federated:latest -f part2-mlops/docker/Dockerfile.server .
docker build -t health-risk-federated-client:latest -f part2-mlops/docker/Dockerfile.client .
docker build -t health-risk-federated-inference:latest -f part2-mlops/docker/Dockerfile.inference .
docker build -t health-risk-federated-mlflow:latest -f part2-mlops/docker/Dockerfile.mlflow part2-mlops
```

#### 2. Start Services with Docker Compose

**Important: Always run from project root!**

```bash
# From project root
docker-compose -f part2-mlops/docker-compose.yml up -d
```

**Verify services:**
```bash
docker-compose -f part2-mlops/docker-compose.yml ps
```

All services should show "Up" status.

#### 3. Train and Register Model

```bash
# Train model and register with MLflow
python part2-mlops/train_model_simple.py

# Restart inference service to load model
docker-compose -f part2-mlops/docker-compose.yml restart inference
```

#### 4. Verify Everything Works

```bash
# Check MLflow
# Open browser: http://localhost:5000

# Check Inference API
curl http://localhost:8080/health
# Should return: {"status":"healthy","model_loaded":true}

# Check Prometheus
# Open browser: http://localhost:9090
```

---

## Usage Examples

### Part 1: Federated Learning

#### Run Federated Training

```bash
# Server (Terminal 1)
python part1-data-model/run_federated.py server

# Client 1 (Terminal 2)
python part1-data-model/run_federated.py client 01

# Client 2 (Terminal 3)
python part1-data-model/run_federated.py client 02
```

#### Generate Data

```python
from data_simulation.wearables import WearableSimulator
from data_simulation.air_quality import EnvironmentalSimulator

# Wearable data
wear_sim = WearableSimulator(num_patients=500)
health_data = wear_sim.generate_daily_data("2024-01-15", "hospital_01")

# Environmental data
env_sim = EnvironmentalSimulator(num_sensors=20)
env_data = env_sim.generate_sensor_data("hospital_01")
```

#### Check Data Drift

```python
from federated_learning.drift_detector import DriftMonitor
import pandas as pd

# Reference data
ref_data = pd.DataFrame({...})

# Monitor
monitor = DriftMonitor(ref_data)

# Check drift
drift = monitor.check_drift(current_data, threshold=0.5)
```

### Part 2: MLOps Operations

#### Make Predictions

**PowerShell:**
```powershell
$body = @{
    heart_rate = 75
    steps = 8000
    sleep_hours = 7.5
    respiratory_rate = 16
    body_temp = 98.6
    pm25 = 15
    pm10 = 25
    o3 = 0.05
    no2 = 0.02
    temperature = 72
    humidity = 50
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
```

**Bash/curl:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "heart_rate": 75,
    "steps": 8000,
    "sleep_hours": 7.5,
    "respiratory_rate": 16,
    "body_temp": 98.6,
    "pm25": 15,
    "pm10": 25,
    "o3": 0.05,
    "no2": 0.02,
    "temperature": 72,
    "humidity": 50
  }'
```

#### Batch Predictions

```powershell
$batch = @(
    @{heart_rate=70; steps=10000; sleep_hours=8; respiratory_rate=15; body_temp=98.4; pm25=10; pm10=20; o3=0.03; no2=0.01; temperature=70; humidity=45},
    @{heart_rate=85; steps=5000; sleep_hours=6; respiratory_rate=18; body_temp=98.8; pm25=35; pm10=50; o3=0.08; no2=0.05; temperature=80; humidity=65}
) | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri http://localhost:8080/predict/batch -Method POST -Body $batch -ContentType "application/json"
```

#### Trigger Retraining

```bash
# Drift detection and retraining
python part2-mlops/mlops/retraining_pipeline.py \
  --mode drift \
  --reference-date 2024-01-14 \
  --current-date 2024-01-15

# Scheduled retraining
python part2-mlops/mlops/retraining_pipeline.py --mode scheduled
```

#### View MLflow Experiments

1. Open http://localhost:5000
2. Navigate to "Experiments"
3. Click on "federated_health_risk"
4. View runs, metrics, parameters, and artifacts

#### Query Prometheus Metrics

1. Open http://localhost:9090
2. Try queries:
   - `predictions_total` - Total predictions
   - `prediction_latency_seconds` - Prediction latency
   - `rate(predictions_total[5m])` - Prediction rate

---

## Troubleshooting

### Common Issues

#### 1. Docker Build Context Error

**Error:** `resolve : CreateFile ... part2-mlops: The system cannot find the file specified`

**Solution:** Always run docker-compose from project root:
```bash
# ✅ Correct
cd /path/to/health-risk-federated
docker-compose -f part2-mlops/docker-compose.yml up -d

# ❌ Wrong
cd part2-mlops
docker-compose up -d
```

#### 2. MLflow Image Not Found

**Error:** `mlflow/mlflow:latest: not found`

**Solution:** Custom MLflow image is built automatically. If needed:
```bash
docker build -t health-risk-federated-mlflow:latest -f part2-mlops/docker/Dockerfile.mlflow part2-mlops
```

#### 3. Port Already in Use

**Error:** `ports are not available: bind: Only one usage of each socket address`

**Solution:**
```powershell
# Find process using port (Windows)
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Or change port in docker-compose.yml
```

#### 4. Model Not Loading

**Error:** `Model not fitted yet`

**Solution:**
1. Train model: `python part2-mlops/train_model_simple.py`
2. Restart inference: `docker-compose -f part2-mlops/docker-compose.yml restart inference`
3. Check MLflow has registered models: http://localhost:5000

#### 5. Clients Cannot Connect to Server

**Error:** Connection refused or timeout

**Solution:**
```bash
# Check server is running
docker-compose -f part2-mlops/docker-compose.yml ps server

# Check SERVER_ADDRESS environment variable
docker-compose -f part2-mlops/docker-compose.yml exec client01 env | grep SERVER_ADDRESS
# Should be: SERVER_ADDRESS=server:5050

# Restart services
docker-compose -f part2-mlops/docker-compose.yml restart server client01 client02
```

#### 6. Permission Denied on Scripts

**Error:** Permission denied when running .sh scripts

**Solution:**
```bash
# Make executable
chmod +x part2-mlops/scripts/*.sh

# Or use bash explicitly
bash part2-mlops/scripts/build-images.sh
```

### Diagnostic Commands

```bash
# Check all services
docker-compose -f part2-mlops/docker-compose.yml ps

# View logs
docker-compose -f part2-mlops/docker-compose.yml logs
docker-compose -f part2-mlops/docker-compose.yml logs [service-name]

# Check Docker images
docker images | grep health-risk

# Check network
docker network ls
docker network inspect part2-mlops_default

# Restart all services
docker-compose -f part2-mlops/docker-compose.yml restart

# Rebuild and restart
docker-compose -f part2-mlops/docker-compose.yml build --no-cache
docker-compose -f part2-mlops/docker-compose.yml up -d
```

---

## Deployment Options

### Option 1: Local Development (Docker Compose)

**Best for:** Development and testing

```bash
# Start all services
docker-compose -f part2-mlops/docker-compose.yml up -d

# Stop services
docker-compose -f part2-mlops/docker-compose.yml down
```

**Services:**
- MLflow: http://localhost:5000
- Inference: http://localhost:8080
- Prometheus: http://localhost:9090

### Option 2: Kubernetes Deployment

**Best for:** Production

**Prerequisites:**
- Kubernetes cluster (minikube, kind, or cloud K8s)
- kubectl configured

**Deploy:**

```bash
# Build images first
./part2-mlops/scripts/build-images.sh

# Deploy to Kubernetes
./part2-mlops/scripts/deploy-k8s.sh

# Or manually:
kubectl apply -f part2-mlops/k8s/namespace.yaml
kubectl apply -f part2-mlops/k8s/configmap.yaml
kubectl apply -f part2-mlops/k8s/mlflow-deployment.yaml
kubectl apply -f part2-mlops/k8s/federated-server-deployment.yaml
kubectl apply -f part2-mlops/k8s/federated-client-deployment.yaml
kubectl apply -f part2-mlops/k8s/inference-deployment.yaml
kubectl apply -f part2-mlops/k8s/prometheus-deployment.yaml
```

**Access Services:**

```bash
# Port forward
kubectl port-forward svc/mlflow-service 5000:5000 -n health-risk-mlops
kubectl port-forward svc/inference-service 8080:8080 -n health-risk-mlops
kubectl port-forward svc/prometheus-service 9090:9090 -n health-risk-mlops
```

### Option 3: CI/CD Pipeline

**Best for:** Automated deployments

The GitHub Actions workflow (`.github/workflows/mlops-ci-cd.yml`) provides:

- **On Push to `develop`**: Deploys to staging
- **On Push to `main`**: Builds images
- **Scheduled**: Drift monitoring every 6 hours
- **Manual**: Training trigger

**Configure:**

1. Set GitHub Secrets:
   - `MLFLOW_TRACKING_URI`
   - `KUBE_CONFIG` (for K8s deployment)

2. Push to trigger workflows

---

## API Reference

### Inference API Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "heart_rate": 75,
  "steps": 8000,
  "sleep_hours": 7.5,
  "respiratory_rate": 16,
  "body_temp": 98.6,
  "pm25": 15,
  "pm10": 25,
  "o3": 0.05,
  "no2": 0.02,
  "temperature": 72,
  "humidity": 50
}
```

**Response:**
```json
{
  "risk_score": 0.25,
  "risk_probability": 0.25,
  "prediction": "low"
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

[
  { ... },
  { ... }
]
```

#### Prometheus Metrics
```http
GET /metrics
```

Returns Prometheus-formatted metrics.

---

## Environment Variables

### Part 1

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://localhost:5000)
- `SERVER_ADDRESS`: Federated server address (default: 127.0.0.1:5050)

### Part 2

**Docker Compose:**
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `SERVER_ADDRESS`: Federated server address (for clients)
- `CLIENT_ID`: Client identifier
- `MODEL_VERSION`: Model version to load

**Kubernetes (ConfigMap):**
- `MLFLOW_TRACKING_URI`: http://mlflow-service:5000
- `DRIFT_THRESHOLD`: 0.5
- `NUM_ROUNDS`: 10
- `MIN_CLIENTS`: 2

---

## Project Structure

```
health-risk-federated/
├── part1-data-model/          # Part 1: Data & Model
│   ├── data_simulation/       # Data generators
│   ├── federated_learning/    # FL implementation
│   ├── models/                # ML models
│   └── run_federated.py       # Main script
├── part2-mlops/               # Part 2: MLOps Pipeline
│   ├── docker/                # Dockerfiles
│   ├── k8s/                   # Kubernetes manifests
│   ├── mlops/                 # MLOps utilities
│   ├── scripts/               # Helper scripts
│   └── docker-compose.yml      # Local setup
├── part3-dashboard/           # Part 3: Dashboard (future)
├── .github/workflows/         # CI/CD pipelines
├── requirements.txt           # Python dependencies
└── COMPLETE_GUIDE.md         # This file
```

---

## Next Steps

### For Development

1. **Integrate with Dashboard**: Connect inference API to Part 3 dashboard
2. **Add More Clients**: Scale federated learning
3. **Improve Model**: Experiment with different algorithms
4. **Add Features**: Extend data simulation

### For Production

1. **Configure Secrets**: Use Kubernetes Secrets for sensitive data
2. **Set Up Ingress**: Configure ingress for external access
3. **Add Alerting**: Set up Prometheus alerts
4. **Backup Strategy**: Regular backups of MLflow data
5. **Monitoring**: Enhanced logging and monitoring
6. **Security**: Network policies, authentication

---

## Support & Resources

### Documentation

- **Flower (Federated Learning)**: https://flower.dev/
- **MLflow**: https://www.mlflow.org/docs/latest/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Prometheus**: https://prometheus.io/docs/

### Getting Help

1. Check logs: `docker-compose -f part2-mlops/docker-compose.yml logs`
2. Review this guide's troubleshooting section
3. Check service status: `docker-compose -f part2-mlops/docker-compose.yml ps`
4. Verify environment variables
5. Check network connectivity

---

## Summary

This project provides a complete MLOps pipeline for federated health risk prediction:

✅ **Part 1**: Data simulation, federated learning, model training  
✅ **Part 2**: Docker, Kubernetes, CI/CD, monitoring, model serving  
✅ **Production Ready**: Scalable, monitored, automated  

**Key Features:**
- Federated learning without data sharing
- Complete MLOps automation
- Model serving via REST API
- Experiment tracking with MLflow
- Monitoring with Prometheus
- Automated retraining

**Status:** ✅ Fully functional and ready for deployment!

---

**Last Updated:** 2024  
**Version:** 1.0

