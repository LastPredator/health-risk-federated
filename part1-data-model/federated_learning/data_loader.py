import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class FederatedHealthDataset(Dataset):
    def __init__(self, health_df, env_df, sequence_length=24):
        self.data = self._merge_data(health_df, env_df)
        self.sequence_length = sequence_length
    
    def _merge_data(self, health_df, env_df):
        """Merge wearable and sensor data on node_id"""
        # FIX: Select only numeric columns for aggregation
        numeric_cols = env_df.select_dtypes(include=[np.number]).columns.tolist()
        # Ensure node_id is included
        if 'node_id' not in numeric_cols:
            numeric_cols = ['node_id'] + numeric_cols
        
        env_numeric = env_df[numeric_cols]
        env_mean = env_numeric.groupby('node_id').mean().reset_index()
        
        merged = health_df.merge(env_mean, on='node_id', how='left')
        return merged.fillna(method='ffill').fillna(0)
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        seq = self.data.iloc[idx:idx+self.sequence_length]
        features = seq[[
            'heart_rate', 'steps', 'sleep_hours', 'respiratory_rate', 'body_temp',
            'pm25', 'pm10', 'o3', 'no2', 'temperature', 'humidity'
        ]].values
        
        label = self.data.iloc[idx+self.sequence_length]['risk_score']
        return torch.FloatTensor(features), torch.LongTensor([label])
    
    def get_dataloader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size, shuffle=True)