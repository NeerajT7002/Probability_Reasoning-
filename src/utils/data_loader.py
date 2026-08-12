import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class SequentialEHRDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        Custom Dataset for sequential electronic health records.
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_mimic_demo(base_path: str, seq_len: int = 24) -> tuple:
    """
    Loads MIMIC-III demo CHARTEVENTS, extracts vitals, and builds sequence tensors.
    
    Args:
        base_path (str): Path to the mimic-iii-clinical-database-demo-1.4 folder
        seq_len (int): Length of output sequence per patient stay (hours)
        
    Returns:
        sequences (np.ndarray): Tensor of shape (num_stays, seq_len, num_features)
        feature_names (list): Names of pivoted vitals features
    """
    chartevents_path = os.path.join(base_path, "CHARTEVENTS.csv")
    
    # Target vitals itemids (Metavision)
    vitals_map = {
        220045: "Heart Rate",
        220179: "Systolic BP",
        220180: "Diastolic BP",
        220210: "Respiratory Rate",
        220277: "SpO2",
        223761: "Temperature"
    }
    
    print("Loading MIMIC-III demo CHARTEVENTS.csv...")
    # Load required columns
    df_chunks = pd.read_csv(
        chartevents_path,
        usecols=["icustay_id", "itemid", "charttime", "valuenum"],
        chunksize=100000
    )
    
    # Filter chunks
    filtered_chunks = []
    for chunk in df_chunks:
        filtered = chunk[chunk["itemid"].isin(vitals_map.keys())].dropna()
        filtered_chunks.append(filtered)
        
    df_vitals = pd.concat(filtered_chunks, axis=0)
    df_vitals["feature"] = df_vitals["itemid"].map(vitals_map)
    
    # Parse timestamps
    df_vitals["charttime"] = pd.to_datetime(df_vitals["charttime"])
    
    # Round timestamps to nearest hour
    df_vitals["hour"] = df_vitals["charttime"].dt.round("h")
    
    print("Pivoting and aggregating hourly clinical measurements...")
    # Group and aggregate by mean hourly value
    df_hourly = df_vitals.groupby(["icustay_id", "hour", "feature"])["valuenum"].mean().unstack(level="feature")
    
    # Reindex columns to guarantee consistent order
    feature_names = ["Heart Rate", "Systolic BP", "Diastolic BP", "Respiratory Rate", "SpO2", "Temperature"]
    df_hourly = df_hourly.reindex(columns=feature_names)
    
    sequences = []
    stay_ids = df_hourly.index.get_level_values("icustay_id").unique()
    
    print(f"Building clinical sequences for {len(stay_ids)} ICU stays...")
    for stay_id in stay_ids:
        stay_data = df_hourly.loc[stay_id]
        
        # Sort chronologically
        stay_data = stay_data.sort_index()
        
        # Forward fill and backward fill missing hourly values
        stay_data = stay_data.ffill().bfill()
        
        # Extract features
        features = stay_data.values
        
        if len(features) >= seq_len:
            # Take first seq_len hours of ICU stay
            seq = features[:seq_len]
        else:
            # Pad suffix with the last measured value or zeros if empty
            pad_width = seq_len - len(features)
            if len(features) > 0:
                seq = np.pad(features, ((0, pad_width), (0, 0)), mode="edge")
            else:
                seq = np.zeros((seq_len, len(feature_names)))
                
        sequences.append(seq)
        
    sequences_arr = np.array(sequences)
    
    # Replace remaining NaNs (if any stay had no values at all) with overall column means
    for f_idx in range(sequences_arr.shape[-1]):
        col = sequences_arr[:, :, f_idx]
        mean_val = np.nanmean(col) if not np.isnan(col).all() else 0.0
        sequences_arr[:, :, f_idx] = np.nan_to_num(col, nan=mean_val)
        
    return sequences_arr, feature_names


def get_dataloader(data: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """
    Generates a PyTorch DataLoader from preprocessed sequence matrices.
    """
    dataset = SequentialEHRDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
