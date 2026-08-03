# Implementation Plan - Replicating DG-HMM

Replicate the **DG-HMM** (Deep Generative Hidden Markov Model) architecture to generate high-fidelity, sequential, mixed-type synthetic health record datasets.

## User Review Required

> [!IMPORTANT]
> **Dataset Access**: The paper evaluates the model on the MIMIC-III database, which requires a CITI training certificate and credentialed access. If you do not have credentialed access to MIMIC-III, we should utilize the **NHANES Diabetes** or a synthetic electronic health record dataset (e.g., Synthea-generated sequences) to test and run our implementation. Please specify which dataset you would like to begin with.

> [!NOTE]
> **HMM Differentiability**: Training the encoder, decoder, and HMM layers jointly requires implementing the HMM Forward algorithm in a fully vectorized and differentiable manner in PyTorch so gradients can flow from the HMM likelihood loss back into the neural network layers.

## Open Questions

- **Dataset Selection**: Do you already have access to the MIMIC-III files (specifically `CHARTEVENTS` or pre-extracted sequence matrices), or should we prepare the pipeline first using public NHANES data or generated/mock EHR trajectories?
- **GPU Availability**: The paper mentions training on an NVIDIA RTX 4060. Do you have a local GPU setup, or will we configure it to run efficiently on CPU/CUDA depending on what is available?

---

## Proposed Changes

### Project Setup and Dependencies

#### [NEW] [requirements.txt](file:///d:/Probability project/DG-HMM/requirements.txt)
Define core dependencies including:
- `torch` (for neural networks and automatic differentiation)
- `numpy`, `pandas`, `scipy` (for data handling and statistical fidelity tests)
- `scikit-learn` (for downstream utility training)
- `fastdtw` (for dynamic time warping calculations)

---

### Core Model Components (`src/models/`)

#### [NEW] [encoder.py](file:///d:/Probability project/DG-HMM/src/models/encoder.py)
Implement the `DeepEncoder` class:
- Maps sequence observations $x_t \in \mathbb{R}^d$ to latent distributions $\mu_z, \log\sigma_z \in \mathbb{R}^k$.
- Applies Batch Normalization and Dropout.
- Implements the reparameterization trick to sample $z_t$.

#### [NEW] [decoder.py](file:///d:/Probability project/DG-HMM/src/models/decoder.py)
Implement the `DeepDecoder` class:
- Takes the concatenated tensor $[z_t; e_{s_t}]$ where $e_{s_t}$ is the one-hot encoded HMM state.
- Reconstructs $\tilde{x}_t \in \mathbb{R}^d$.
- Supports split outputs if the variables are mixed-type (continuous, categorical, binary, count).

#### [NEW] [hmm.py](file:///d:/Probability project/DG-HMM/src/models/hmm.py)
Implement the `DifferentiableHMM` layer:
- Tracks transition probability matrix $A$, initial state distribution $\pi$, and emission models.
- Implements the forward algorithm using PyTorch tensors for calculating log-likelihood $\mathcal{L}_{HMM} = -\log p(\mathbf{Z} | \theta_{HMM})$.
- Implements state-dependent GMM or Normalizing Flow emissions.

#### [NEW] [dghmm.py](file:///d:/Probability project/DG-HMM/src/models/dghmm.py)
Implement the top-level `DGHMM` class:
- Coordinates the encoder, HMM layer, and decoder.
- Implements `forward` pass to compute joint reconstruction, KL-divergence, and HMM likelihood losses.
- Implements synthetic sample generation (`generate(num_samples, seq_length)`).

---

### Data Pipeline and Utilities (`src/utils/`)

#### [NEW] [flows.py](file:///d:/Probability project/DG-HMM/src/utils/flows.py)
Implement auxiliary probability density estimators:
- Mixture of Gaussians (GMM) class in PyTorch.
- Simple Normalizing Flow (e.g., RealNVP or Planar flows) for mapping complex latent states.

#### [NEW] [data_loader.py](file:///d:/Probability project/DG-HMM/src/utils/data_loader.py)
Provide loaders and preprocessing scripts:
- Clean and normalize continuous values.
- Pad sequences to uniform length.
- Convert raw tabular CSV records into sequence tensors.

---

### Validation Suite (`src/evaluation/`)

#### [NEW] [fidelity.py](file:///d:/Probability project/DG-HMM/src/evaluation/fidelity.py)
Compute statistical metrics:
- Kolmogorov-Smirnov (KS) statistic.
- Pearson correlation matrix comparisons.

#### [NEW] [temporal.py](file:///d:/Probability project/DG-HMM/src/evaluation/temporal.py)
Compute sequential consistency:
- Autocorrelation Function (ACF) lag performance.
- Dynamic Time Warping (DTW) distance between real and synthetic patient records.

#### [NEW] [privacy.py](file:///d:/Probability project/DG-HMM/src/evaluation/privacy.py)
Compute privacy preservation metrics:
- Membership Inference Attack (MIA) simulation.
- k-anonymity score estimation.

---

## Verification Plan

### Automated Tests
We will write a synthetic dataset generator test script:
- Create mock sequences with continuous/binary dimensions.
- Run a 1-epoch training loop to verify that gradients flow from all losses (`L_recon`, `L_hmm`, `L_reg`) to both the neural network weights and the HMM transition matrices.
- Command to run:
  ```bash
  python -m unittest discover -s tests
  ```

### Manual Verification
- Execute a training run on a mock sequence dataset to verify that reconstructed sequences match the temporal profile of the input.
- Run synthetic data generation and export samples to CSV to visually inspect state transitions.
