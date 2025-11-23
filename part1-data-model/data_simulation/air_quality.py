import pandas as pd
import numpy as np
from datetime import datetime

class EnvironmentalSimulator:
    def __init__(self, num_sensors=20):
        self.sensor_ids = [f"SENSOR_{i:02d}" for i in range(num_sensors)]
        self.cities = ['NYC', 'LA', 'Chicago', 'Houston']
    
    def generate_sensor_data(self, node_id="city_01"):
        """Generate environmental data for a node"""
        return pd.DataFrame({
            'node_id': node_id,
            'sensor_id': self.sensor_ids,
            'city': np.random.choice(self.cities, len(self.sensor_ids)),
            'timestamp': datetime.now(),
            'pm25': np.random.exponential(12, len(self.sensor_ids)),
            'pm10': np.random.exponential(20, len(self.sensor_ids)),
            'o3': np.random.normal(0.035, 0.01, len(self.sensor_ids)),
            'no2': np.random.exponential(18, len(self.sensor_ids)),
            'temperature': np.random.normal(70, 12, len(self.sensor_ids)),
            'humidity': np.random.normal(55, 15, len(self.sensor_ids))
        })
    
    def get_weather_forecast(self, node_id="city_01"):
        """Simulate weather data"""
        return {
            'node_id': node_id,
            'temperature_high': np.random.normal(82, 8),
            'air_quality_index': np.random.randint(40, 180),
            'uv_index': np.random.randint(1, 11),
            'pollen_level': np.random.choice(['low', 'medium', 'high'])
        }