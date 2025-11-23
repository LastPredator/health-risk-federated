"""
Simple script to train a model and register it with MLflow
Run this to get a working model for the inference API
"""
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../part1-data-model'))

from models.health_risk_model import HealthRiskModel
from data_simulation.wearables import WearableSimulator
from data_simulation.air_quality import EnvironmentalSimulator
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

def train_and_register_model():
    """Train a simple model and register it with MLflow"""
    
    print("Training health risk model...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("federated_health_risk")
    
    # Generate training data
    print("Generating training data...")
    wear_sim = WearableSimulator(num_patients=500)
    env_sim = EnvironmentalSimulator(num_sensors=20)
    
    health_data = wear_sim.generate_daily_data("2024-01-15", node_id="hospital_01")
    env_data = env_sim.generate_sensor_data(node_id="hospital_01")
    
    # Merge data (only numeric columns)
    numeric_cols = env_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'node_id' not in numeric_cols:
        numeric_cols = ['node_id'] + numeric_cols
    env_numeric = env_data[numeric_cols]
    env_mean = env_numeric.groupby('node_id').mean().reset_index()
    merged = health_data.merge(env_mean, on='node_id', how='left').fillna(0)
    
    # Prepare features and labels
    feature_cols = [
        'heart_rate', 'steps', 'sleep_hours', 'respiratory_rate', 'body_temp',
        'pm25', 'pm10', 'o3', 'no2', 'temperature', 'humidity'
    ]
    
    X = merged[feature_cols].values
    y = merged['risk_score'].values
    
    print(f"Generated {len(X)} training samples")
    
    # Train model
    print("Training model...")
    model = HealthRiskModel()
    model.fit(X, y)
    
    # Test prediction
    test_pred = model.predict_proba(X[:5])
    print(f"Model trained! Test predictions: {test_pred[:, 1]}")
    
    # Register with MLflow
    print("Registering model with MLflow...")
    with mlflow.start_run(run_name="simple_training"):
        # Save the full HealthRiskModel (including scaler) as artifact
        import joblib
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "health_risk_model.pkl")
            model.save(model_path)
            
            # Log the full model as artifact
            mlflow.log_artifact(model_path, "model")
            
            # Also log just the sklearn model for compatibility
            mlflow.sklearn.log_model(
                model.model,
                "sklearn_model",
                registered_model_name="health_risk_model"
            )
        
        # Log parameters
        mlflow.log_param("training_samples", len(X))
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("training_method", "simple")
        
        # Calculate and log metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model registered!")
        print(f"   AUC: {auc:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   MLflow UI: http://localhost:5000")
    
    print("\nModel training complete!")
    print("   The inference API should now be able to load this model.")
    print("   Restart the inference service to load the new model:")
    print("   docker-compose -f part2-mlops/docker-compose.yml restart inference")

if __name__ == "__main__":
    train_and_register_model()

