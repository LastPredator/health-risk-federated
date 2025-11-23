# Part 2: MLOps Pipeline

For complete instructions covering both Part 1 and Part 2, see the **[COMPLETE_GUIDE.md](../COMPLETE_GUIDE.md)** in the project root.

This directory contains the MLOps infrastructure for the Health Risk Federated Learning project.

## Quick Reference

### Start Services
```bash
# From project root
docker-compose -f part2-mlops/docker-compose.yml up -d
```

### Train Model
```bash
python part2-mlops/train_model_simple.py
docker-compose -f part2-mlops/docker-compose.yml restart inference
```

### Access Services
- MLflow: http://localhost:5000
- Inference API: http://localhost:8080
- Prometheus: http://localhost:9090

For detailed instructions, see [COMPLETE_GUIDE.md](../COMPLETE_GUIDE.md).
