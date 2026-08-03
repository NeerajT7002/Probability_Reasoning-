import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_autocorrelation(data: np.ndarray, lag: int = 1) -> float:
    """
    Computes autocorrelation at a given lag for all features across time.
    
    Args:
        data (np.ndarray): Sequential data of shape (num_samples, seq_len, num_features)
        lag (int): Autocorrelation lag steps.
        
    Returns:
        mean_acf (float): Average autocorrelation coefficient.
    """
    num_samples, seq_len, num_features = data.shape
    if seq_len <= lag:
        return 0.0
        
    acfs = []
    for s in range(num_samples):
        for f in range(num_features):
            series = data[s, :, f]
            mean = np.mean(series)
            var = np.var(series) + 1e-8
            
            # Autocovariance
            cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acfs.append(cov / var)
            
    return float(np.mean(acfs))


def compute_dtw_distance(real_data: np.ndarray, synthetic_data: np.ndarray, max_samples: int = 100) -> float:
    """
    Calculates average Dynamic Time Warping (DTW) distance between sequence pairs.
    
    Args:
        real_data (np.ndarray): Real sequences of shape (num_samples, seq_len, num_features)
        synthetic_data (np.ndarray): Synthetic sequences of shape (num_samples, seq_len, num_features)
        max_samples (int): Cap comparison pairs for performance optimization.
        
    Returns:
        mean_dtw (float): Average DTW distance.
    """
    n_real = len(real_data)
    n_synth = len(synthetic_data)
    
    # Cap samples to keep computation time fast
    num_comparisons = min(max_samples, n_real, n_synth)
    
    dtw_distances = []
    for i in range(num_comparisons):
        # Compute DTW between pair (real[i], synthetic[i])
        dist, _ = fastdtw(real_data[i], synthetic_data[i], dist=euclidean)
        dtw_distances.append(dist)
        
    return float(np.mean(dtw_distances))
