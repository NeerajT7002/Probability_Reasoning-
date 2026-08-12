import numpy as np
from scipy.stats import ks_2samp

def compute_ks_score(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes average Kolmogorov-Smirnov (KS) statistic across all feature dimensions.
    
    Args:
        real_data (np.ndarray): Real sequences of shape (num_samples, seq_len, num_features)
        synthetic_data (np.ndarray): Synthetic sequences of shape (num_samples, seq_len, num_features)
        
    Returns:
        average_ks (float): Average KS statistic score.
    """
    # Flatten across batch and time dimensions to evaluate global distribution fidelity
    num_features = real_data.shape[-1]
    real_flat = real_data.reshape(-1, num_features)
    synthetic_flat = synthetic_data.reshape(-1, num_features)
    
    ks_stats = []
    for i in range(num_features):
        stat, _ = ks_2samp(real_flat[:, i], synthetic_flat[:, i])
        ks_stats.append(stat)
        
    return float(np.mean(ks_stats))


def compute_correlation_preservation(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Measures the preservation of inter-variable pairwise Pearson correlations.
    
    Args:
        real_data (np.ndarray): Real sequences of shape (num_samples, seq_len, num_features)
        synthetic_data (np.ndarray): Synthetic sequences of shape (num_samples, seq_len, num_features)
        
    Returns:
        corr_preservation (float): Pearson correlation between the real and synthetic correlation matrices.
    """
    num_features = real_data.shape[-1]
    real_flat = real_data.reshape(-1, num_features)
    synthetic_flat = synthetic_data.reshape(-1, num_features)
    
    # Calculate correlation matrices
    corr_real = np.nan_to_num(np.corrcoef(real_flat, rowvar=False))
    corr_synth = np.nan_to_num(np.corrcoef(synthetic_flat, rowvar=False))
    
    # Flatten upper triangular matrix elements to avoid double counting or self-correlation
    triu_indices = np.triu_indices(num_features, k=1)
    vector_real = corr_real[triu_indices]
    vector_synth = corr_synth[triu_indices]
    
    # Compute Pearson correlation between the two vector sets
    overall_corr = np.corrcoef(vector_real, vector_synth)[0, 1]
    return float(np.nan_to_num(overall_corr))
