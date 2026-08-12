import os
import torch
import numpy as np
from src.utils.data_loader import preprocess_mimic_demo, get_dataloader
from src.models.dghmm import DGHMM
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import compute_autocorrelation, compute_dtw_distance
from src.evaluation.privacy import evaluate_mia_resistance, compute_k_anonymity

def main():
    # 1. Device Setup (Leverage GPU RTX 2050)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    if device.type == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    # 2. Paths and Parameters
    base_path = r"d:\Probability project\DG-HMM\mimic-iii-clinical-database-demo-1.4"
    seq_len = 24  # 24-hour window
    latent_dim = 8
    num_states = 4
    epochs = 20
    batch_size = 8
    
    # 3. Data Preprocessing
    print("\n--- Data Preprocessing ---")
    data, feature_names = preprocess_mimic_demo(base_path, seq_len=seq_len)
    print(f"Pivoted Vitals Features: {feature_names}")
    print(f"Processed Sequence Matrix Shape: {data.shape}")
    
    # Train / Test split
    np.random.seed(42)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(data))
    
    train_data = data[indices[:split_idx]]
    test_data = data[indices[split_idx:]]
    print(f"Training cases: {len(train_data)}, Testing cases: {len(test_data)}")
    
    train_loader = get_dataloader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True)
    
    # 4. Initialize DG-HMM Model
    input_dim = data.shape[-1]
    model = DGHMM(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_states=num_states
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 5. Training Loop
    print("\n--- Training DG-HMM ---")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Compute joint loss
            loss, loss_details = model.compute_loss(batch, state_mode='viterbi')
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_details['loss_total'])
            
        mean_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else 0.0
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Total Joint Loss: {mean_loss:.4f} | Recon: {loss_details['loss_recon']:.4f} | HMM: {loss_details['loss_hmm']:.4f}")

    # 6. Generate Synthetic Clinical Patient Stays
    print("\n--- Generating Synthetic Vitals Data ---")
    num_synthetic = 50
    synth_data_torch, synth_states_torch = model.generate(num_samples=num_synthetic, seq_len=seq_len, device=device)
    
    synth_data = synth_data_torch.cpu().numpy()
    synth_states = synth_states_torch.cpu().numpy()
    print(f"Generated Synthetic Sequences Shape: {synth_data.shape}")
    
    # 7. Evaluate Performance
    print("\n--- Evaluation Metrics ---")
    
    # Statistical Fidelity
    ks_score = compute_ks_score(data, synth_data)
    corr_pres = compute_correlation_preservation(data, synth_data)
    print(f"Kolmogorov-Smirnov (KS) Score (lower is better): {ks_score:.4f}")
    print(f"Correlation Matrix Preservation (higher is better): {corr_pres:.4f}")
    
    # Temporal Consistency
    real_acf = compute_autocorrelation(data, lag=1)
    synth_acf = compute_autocorrelation(synth_data, lag=1)
    dtw_dist = compute_dtw_distance(data, synth_data, max_samples=30)
    print(f"Real Data Lag-1 Autocorrelation: {real_acf:.4f}")
    print(f"Synthetic Data Lag-1 Autocorrelation: {synth_acf:.4f}")
    print(f"Dynamic Time Warping (DTW) Distance: {dtw_dist:.4f}")
    
    # Privacy Protection
    mia_res = evaluate_mia_resistance(train_data, test_data, synth_data)
    k_anon = compute_k_anonymity(synth_data)
    print(f"Membership Inference Attack (MIA) Resistance (target ~1.0): {mia_res:.4f}")
    print(f"k-Anonymity Class Score: {k_anon}")
    
    # Print sample generated sequence values
    print("\nSample Generated Sequence (First 5 hours of first synthetic patient):")
    for hour in range(5):
        vals = [f"{feature_names[i]}: {synth_data[0, hour, i]:.2f}" for i in range(input_dim)]
        print(f"Hour {hour + 1} (State {synth_states[0, hour]}): {', '.join(vals)}")

if __name__ == "__main__":
    main()
