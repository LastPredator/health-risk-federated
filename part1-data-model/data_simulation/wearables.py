import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WearableSimulator:
    def __init__(self, num_patients=500):
        self.patient_ids = [f"PT_{i:04d}" for i in range(num_patients)]
    
    def generate_daily_data(self, date, node_id="hospital_01"):
        """Generate one day of wearable data for a specific node"""
        # Base health metrics
        base_hr = np.random.normal(72, 8, len(self.patient_ids))
        base_steps = np.random.exponential(6000, len(self.patient_ids))
        base_sleep = np.random.normal(7, 1.2, len(self.patient_ids))
        
        # Inject health risks for 15% of patients
        high_risk_mask = np.random.random(len(self.patient_ids)) > 0.85
        base_hr[high_risk_mask] += np.random.normal(18, 5, sum(high_risk_mask))
        base_sleep[high_risk_mask] -= np.random.normal(2, 0.5, sum(high_risk_mask))
        
        return pd.DataFrame({
            'node_id': node_id,
            'patient_id': self.patient_ids,
            'timestamp': pd.Timestamp(date),
            'heart_rate': np.clip(base_hr, 40, 160),
            'steps': np.clip(base_steps, 0, 30000).astype(int),
            'sleep_hours': np.clip(base_sleep, 0, 12),
            'respiratory_rate': np.random.normal(16, 2, len(self.patient_ids)),
            'body_temp': np.random.normal(98.6, 0.6, len(self.patient_ids)),
            'risk_score': high_risk_mask.astype(int)
        })