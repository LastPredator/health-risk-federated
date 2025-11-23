import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WeatherSimulator:
    def __init__(self, locations=['NYC', 'LA', 'Chicago', 'Houston']):
        self.locations = locations
    
    def generate_forecast(self, node_id="city_01"):
        """Generate simulated weather forecast data"""
        return {
            'node_id': node_id,
            'city': np.random.choice(self.locations),
            'timestamp': datetime.now(),
            'temperature_high': float(np.random.normal(82, 8)),
            'temperature_low': float(np.random.normal(62, 6)),
            'air_quality_index': int(np.random.randint(40, 180)),
            'uv_index': int(np.random.randint(1, 11)),
            'pollen_level': np.random.choice(['low', 'medium', 'high', 'very_high']),
            'humidity': float(np.random.normal(55, 15)),
            'wind_speed': float(np.random.exponential(8))
        }
    
    def generate_historical_data(self, node_id="city_01", days=30):
        """Generate historical weather data for the past N days"""
        dates = pd.date_range(end=datetime.now(), periods=days)
        data = []
        
        for date in dates:
            data.append({
                'node_id': node_id,
                'date': date,
                'temperature': float(np.random.normal(75, 10)),
                'humidity': float(np.random.normal(55, 15)),
                'precipitation': float(np.random.exponential(0.1)),
                'air_pressure': float(np.random.normal(1013, 5))
            })
        
        return pd.DataFrame(data)