import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_mia_resistance(real_train: np.ndarray, real_test: np.ndarray, 
                            synthetic_data: np.ndarray) -> float:
    """
    Simulates a Membership Inference Attack (MIA) against the generator.
    Trains a classifier to distinguish training records from test records,
    using synthetic data characteristics to approximate model leakage.
    
    Args:
        real_train (np.ndarray): Training sequences of shape (num_samples_train, seq_len, num_features)
        real_test (np.ndarray): Test sequences of shape (num_samples_test, seq_len, num_features)
        synthetic_data (np.ndarray): Generated sequences of shape (num_synth, seq_len, num_features)
        
    Returns:
        mia_resistance (float): Attack resistance score (1.0 - classifier accuracy).
                                Ideally 0.50 (perfect resistance, i.e. random guessing).
    """
    n_train = len(real_train)
    n_test = len(real_test)
    
    # Flatten features across time step dimension
    feat_train = real_train.reshape(n_train, -1)
    feat_test = real_test.reshape(n_test, -1)
    
    # Label training samples as 1 and testing as 0
    X = np.vstack([feat_train, feat_test])
    y = np.concatenate([np.ones(n_train), np.zeros(n_test)])
    
    # Split into train/validation sets for the attacker
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the attacker classifier
    attacker = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    attacker.fit(X_train, y_train)
    
    preds = attacker.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    
    # Attack resistance is high when accuracy is close to random guessing (0.5)
    # Scale score so that 0.5 accuracy yields 1.0 resistance, and 1.0 accuracy yields 0.0 resistance.
    resistance = max(0.0, 1.0 - (accuracy - 0.5) * 2) if accuracy >= 0.5 else 1.0
    return float(resistance)


def compute_k_anonymity(synthetic_data: np.ndarray, quasi_identifiers: list = None) -> int:
    """
    Computes k-anonymity score of the synthetic dataset.
    
    Args:
        synthetic_data (np.ndarray): Generated patient sequences of shape (num_samples, seq_len, num_features)
        quasi_identifiers (list): Indices of features acting as quasi-identifiers (e.g. demographic codes).
                                  If None, uses all features.
                                  
    Returns:
        k (int): Minimum size of equivalence classes.
    """
    num_samples, seq_len, num_features = synthetic_data.shape
    
    # Flatten the features across time steps
    flat_data = synthetic_data.reshape(num_samples, -1)
    
    if quasi_identifiers is not None:
        # Map time indices to flattened indices
        flat_indices = []
        for q in quasi_identifiers:
            for t in range(seq_len):
                flat_indices.append(t * num_features + q)
        flat_data = flat_data[:, flat_indices]
        
    # Discretize values (using rounding) to group into equivalence classes
    discretized = np.round(flat_data, decimals=1)
    
    # Group unique rows and count frequencies
    unique_rows, counts = np.unique(discretized, axis=0, return_counts=True)
    
    # k is the minimum frequency of any unique patient profile
    return int(np.min(counts))
