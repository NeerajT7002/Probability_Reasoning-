import unittest
import torch
import numpy as np
import pandas as pd
from src.models.encoder import DeepEncoder
from src.models.decoder import DeepDecoder
from src.models.hmm import DifferentiableHMM
from src.models.dghmm import DGHMM
from src.utils.flows import NormalizingFlow
from src.utils.data_loader import preprocess_tabular_to_sequences, get_dataloader
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import compute_autocorrelation, compute_dtw_distance
from src.evaluation.privacy import evaluate_mia_resistance, compute_k_anonymity

class TestDGHMM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 8
        self.latent_dim = 4
        self.num_states = 3
        self.device = 'cpu'
        
        # Create random input sequence
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_encoder(self):
        encoder = DeepEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        z, mu, logvar = encoder(self.x)
        
        self.assertEqual(z.shape, (self.batch_size, self.seq_len, self.latent_dim))
        self.assertEqual(mu.shape, (self.batch_size, self.seq_len, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.seq_len, self.latent_dim))

    def test_decoder(self):
        decoder = DeepDecoder(latent_dim=self.latent_dim, num_states=self.num_states, output_dim=self.input_dim)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        state_onehot = torch.zeros(self.batch_size, self.seq_len, self.num_states)
        state_onehot[:, :, 0] = 1.0  # mock all in state 0
        
        x_recon = decoder(z, state_onehot)
        self.assertEqual(x_recon.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_hmm(self):
        hmm = DifferentiableHMM(num_states=self.num_states, latent_dim=self.latent_dim)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        
        # Test forward loss computation
        loss = hmm.forward_loss(z)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # scalar
        
        # Test viterbi path decoding
        states = hmm.viterbi(z)
        self.assertEqual(states.shape, (self.batch_size, self.seq_len))
        self.assertTrue(states.min() >= 0)
        self.assertTrue(states.max() < self.num_states)

    def test_dghmm_joint(self):
        model = DGHMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_states=self.num_states
        )
        
        # Compute joint loss
        loss, details = model.compute_loss(self.x)
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn('loss_total', details)
        self.assertIn('loss_recon', details)
        self.assertIn('loss_hmm', details)
        self.assertIn('loss_kl', details)
        
        # Test backward pass (gradient flow verification)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient!")

    def test_generation(self):
        model = DGHMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_states=self.num_states
        )
        
        num_gen = 5
        gen_len = 12
        synth_x, synth_states = model.generate(num_samples=num_gen, seq_len=gen_len)
        
        self.assertEqual(synth_x.shape, (num_gen, gen_len, self.input_dim))
        self.assertEqual(synth_states.shape, (num_gen, gen_len))

    def test_flows(self):
        flow = NormalizingFlow(dim=self.latent_dim, num_layers=2)
        z = torch.randn(self.batch_size, self.latent_dim)
        
        transformed, log_det = flow(z)
        self.assertEqual(transformed.shape, z.shape)
        self.assertEqual(log_det.shape, (self.batch_size,))
        
        reconstructed = flow.inverse(transformed)
        self.assertTrue(torch.allclose(z, reconstructed, atol=1e-4))

    def test_data_loader(self):
        data = {
            'patient_id': [1, 1, 1, 2, 2],
            'time_step': [1, 2, 3, 1, 2],
            'feature1': [0.1, 0.2, 0.3, 0.5, 0.6],
            'feature2': [1.0, 2.0, 1.5, 3.0, 4.0]
        }
        df = pd.DataFrame(data)
        sequences = preprocess_tabular_to_sequences(
            df, id_col='patient_id', time_col='time_step',
            feature_cols=['feature1', 'feature2'], seq_len=3
        )
        # Expected shape: (2 patients, 3 steps, 2 features)
        self.assertEqual(sequences.shape, (2, 3, 2))
        
        loader = get_dataloader(sequences, batch_size=2, shuffle=False)
        for batch in loader:
            self.assertEqual(batch.shape, (2, 3, 2))

    def test_evaluation_metrics(self):
        real = np.random.randn(10, 5, 2)
        synth = np.random.randn(10, 5, 2)
        
        # Test fidelity
        ks = compute_ks_score(real, synth)
        corr_pres = compute_correlation_preservation(real, synth)
        self.assertTrue(0.0 <= ks <= 1.0)
        self.assertTrue(-1.0 <= corr_pres <= 1.0)
        
        # Test temporal
        acf = compute_autocorrelation(real, lag=1)
        self.assertTrue(isinstance(acf, float))
        
        # Test privacy
        real_train = real[:6]
        real_test = real[6:]
        mia = evaluate_mia_resistance(real_train, real_test, synth)
        k_anon = compute_k_anonymity(synth)
        self.assertTrue(0.0 <= mia <= 1.0)
        self.assertTrue(k_anon >= 1)

if __name__ == '__main__':
    unittest.main()

