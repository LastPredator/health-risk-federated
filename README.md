# Health Risk Federated Learning - MLOps Project

End-to-end MLOps system for health risk prediction using federated learning.

## 📚 Documentation

**👉 See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) for comprehensive instructions covering both Part 1 and Part 2.**

## Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### Installation
```bash
# Clone repository
git clone https://github.com/LastPredator/health-risk-federated.git
cd health-risk-federated

# Install dependencies
pip install -r requirements.txt
```

### Run Part 1 (Federated Learning)
```bash
# Terminal 1: Server
python part1-data-model/run_federated.py server

# Terminal 2: Client 1
python part1-data-model/run_federated.py client 01

# Terminal 3: Client 2
python part1-data-model/run_federated.py client 02
```

### Run Part 2 (MLOps Pipeline)
```bash
# From project root
docker-compose -f part2-mlops/docker-compose.yml up -d

# Train model
python part2-mlops/train_model_simple.py

# Access services
# MLflow: http://localhost:5000
# Inference API: http://localhost:8080
# Prometheus: http://localhost:9090
```

## Project Structure

- **part1-data-model/**: Data simulation, federated learning, models
- **part2-mlops/**: Docker, Kubernetes, CI/CD, monitoring
- **part3-dashboard/**: Dashboard (future work)

## Features

- ✅ Federated Learning (Flower)
- ✅ Data Simulation (Wearables, Air Quality)
- ✅ Model Training & Tracking (MLflow)
- ✅ Model Serving (FastAPI)
- ✅ Monitoring (Prometheus)
- ✅ CI/CD Pipeline (GitHub Actions)
- ✅ Kubernetes Deployment
- ✅ Automated Retraining

## Documentation

For complete setup, usage, and troubleshooting instructions, see **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)**.

## License

See LICENSE file.

