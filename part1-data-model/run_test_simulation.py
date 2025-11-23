from data_simulation.wearables import WearableSimulator
from data_simulation.air_quality import EnvironmentalSimulator
from data_simulation.weather import WeatherSimulator

# Test all simulators
for i in range(3):
    node_id = f"hospital_{i:02d}"
    
    # Health data
    wear_sim = WearableSimulator(num_patients=100)
    health_data = wear_sim.generate_daily_data("2024-01-15", node_id)
    
    # Environmental data
    env_sim = EnvironmentalSimulator(num_sensors=5)
    env_data = env_sim.generate_sensor_data(node_id)
    
    # Weather data
    weather_sim = WeatherSimulator()
    weather = weather_sim.generate_forecast(node_id)
    
    print(f"\n=== {node_id} ===")
    print(f"Health records: {len(health_data)}")
    print(f"Risk cases: {health_data['risk_score'].sum()}")
    print(f"PM2.5 avg: {env_data['pm25'].mean():.1f}")
    print(f"Weather forecast: {weather['air_quality_index']} AQI")
    print(f"Temperature: {weather['temperature_high']:.1f}°F")