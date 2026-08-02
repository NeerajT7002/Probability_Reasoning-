# Deep Generative Hidden Markov Models (DG-HMM)

An implementation of **DG-HMM** for synthetic patient data generation, replicating the framework from the paper: *"Deep generative hidden Markov models for synthetic patient data generation: a novel approach for medical AI research"* (2026).

## Overview

DG-HMM combines the representational power of deep neural networks (encoder-decoder architectures) with the sequential probabilistic modeling strengths of Hidden Markov Models (HMMs) to generate realistic, sequence-based synthetic electronic health records (EHRs) while preserving patient privacy.

```
       +-----------------------+
       |  Patient Observation  | (x_t)
       +-----------+-----------+
                   |
                   v
       +-----------------------+
       |   Deep MLP Encoder    | (f_phi)
       +-----------+-----------+
                   |
                   v
       +-----------------------+      +-----------------------+
       |  Latent Vector (z_t)  +----->|  HMM State Transition | (s_t)
       +-----------+-----------+      +-----------+-----------+
                   |                              |
                   +--------------+---------------+
                                  |
                                  v [z_t; e_st] (Concatenated)
                       +----------------------+
                       |   Deep MLP Decoder   | (g_psi)
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Synthetic Record     | (~x_t)
                       +----------------------+
```

### Key Features
- **Mixed-Type Emission Support**: Handles continuous, categorical, binary, and count variables simultaneously.
- **Latent Space HMM Layer**: Models temporal state progression using a differentiable forward algorithm implementation.
- **Normalizing Flows & GMMs**: Implements flexible latent density estimation.
- **Joint Optimization**: End-to-end training using a composite loss function comprising reconstruction error, HMM log-likelihood, and VAE-style regularization.

---

## Repository Structure

```
├── data/                  # Placeholder for datasets (MIMIC-III, NHANES, etc.)
├── src/
│   ├── models/            # DG-HMM and baseline model architectures
│   │   ├── __init__.py
│   │   ├── encoder.py     # Deep Encoder network
│   │   ├── decoder.py     # Deep Decoder network
│   │   ├── hmm.py         # Latent HMM layer & Forward algorithm
│   │   └── dghmm.py       # Core DG-HMM orchestrator
│   ├── utils/             # Data loading, preprocessing, and helpers
│   │   ├── data_loader.py
│   │   └── flows.py       # Normalizing flow / GMM helper layers
│   └── evaluation/        # Validation and metric computations
│       ├── fidelity.py    # KS test, Pearson correlation
│       ├── temporal.py    # ACF, DTW
│       └── privacy.py     # MIA, k-anonymity
├── notebooks/             # Exploratory analysis and training demos
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## Installation

To set up the project locally:

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd DG-HMM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Getting Started

*(Quickstart code will be added here upon completion of the implementation phase)*
