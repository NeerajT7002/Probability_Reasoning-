import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SequentialEHRDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 10):
        """
        Custom Dataset for sequential electronic health records.
        
        Args:
            data (np.ndarray): Array of shape (num_samples, seq_len, num_features)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_tabular_to_sequences(df: pd.DataFrame, id_col: str, time_col: str, 
                                    feature_cols: list, seq_len: int = 10) -> np.ndarray:
    """
    Transforms flat tabular records into sequential patient tensors.
    
    Args:
        df (pd.DataFrame): Flat table containing multiple time steps per patient.
        id_col (str): Column mapping patient identity.
        time_col (str): Column defining step sequence ordering.
        feature_cols (list): List of column names to extract as features.
        seq_len (int): Output sequence length per patient.
        
    Returns:
        sequences (np.ndarray): Tensor of shape (num_patients, seq_len, len(feature_cols))
    """
    # Sort data by patient ID and timestamp
    df_sorted = df.sort_values(by=[id_col, time_col])
    
    sequences = []
    
    # Group by patient
    grouped = df_sorted.groupby(id_col)
    
    for pid, group in grouped:
        features = group[feature_cols].values
        
        # Padding or truncating sequence length
        if len(features) >= seq_len:
            # Take last seq_len steps
            seq = features[-seq_len:]
        else:
            # Zero-padding prefix
            pad_width = seq_len - len(features)
            seq = np.pad(features, ((pad_width, 0), (0, 0)), mode='constant', constant_values=0.0)
            
        sequences.append(seq)
        
    return np.array(sequences)


def get_dataloader(data: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """
    Generates a PyTorch DataLoader from preprocessed sequence matrices.
    """
    dataset = SequentialEHRDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
