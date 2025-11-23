from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDrift
import pandas as pd
import mlflow
from data_simulation.wearables import WearableSimulator
from data_simulation.air_quality import EnvironmentalSimulator

class DriftMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.column_mapping = ColumnMapping(
            prediction='risk_prediction',
            target='risk_score',
            numerical_features=['heart_rate', 'steps', 'sleep_hours', 'pm25']
        )
    
    def check_drift(self, current_data: pd.DataFrame, threshold=0.5):
        """Returns True if drift detected"""
        report = Report(metrics=[DataDrift()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        drift_score = report.as_dict()['metrics'][0]['result']['dataset_drift']
        mlflow.log_metric("drift_score", drift_score)
        
        if drift_score > threshold:
            print(f"🚨 DRIFT DETECTED: Score = {drift_score:.2f}")
            return True
        
        print(f"✅ No drift: Score = {drift_score:.2f}")
        return False

# Quick test
if __name__ == "__main__":
    wear_sim = WearableSimulator(100)
    env_sim = EnvironmentalSimulator(5)
    
    ref_data = wear_sim.generate_daily_data("2024-01-14", "hospital_01")
    env_ref = env_sim.generate_sensor_data("hospital_01")
    ref_merged = ref_data.merge(env_ref.groupby('node_id').mean().reset_index(), on='node_id')
    
    curr_data = wear_sim.generate_daily_data("2024-01-15", "hospital_01")
    curr_data['heart_rate'] *= 1.3
    curr_merged = curr_data.merge(env_ref.groupby('node_id').mean().reset_index(), on='node_id')
    
    monitor = DriftMonitor(ref_merged)
    monitor.check_drift(curr_merged, threshold=0.3)