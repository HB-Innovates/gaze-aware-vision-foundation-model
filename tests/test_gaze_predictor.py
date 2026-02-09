"""Unit tests for gaze tracking predictor."""

import pytest
import torch
import numpy as np
from models.gaze_tracking.predictor import GazePredictor, TemporalPredictor


class TestGazePredictor:
    """Test cases for GazePredictor."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return GazePredictor(input_channels=1, hidden_dim=128)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(4, 1, 64, 64)  # batch_size=4
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass produces correct output shape."""
        model.eval()
        with torch.no_grad():
            yaw, pitch = model(sample_input)
        
        assert yaw.shape == (4,), f"Expected shape (4,), got {yaw.shape}"
        assert pitch.shape == (4,), f"Expected shape (4,), got {pitch.shape}"
    
    def test_output_range(self, model, sample_input):
        """Test outputs are in reasonable range."""
        model.eval()
        with torch.no_grad():
            yaw, pitch = model(sample_input)
        
        # Gaze angles typically in [-π, π] range
        assert torch.all(torch.abs(yaw) < 5.0), "Yaw out of reasonable range"
        assert torch.all(torch.abs(pitch) < 5.0), "Pitch out of reasonable range"
    
    def test_gradient_flow(self, model, sample_input):
        """Test gradients can flow through model."""
        model.train()
        yaw, pitch = model(sample_input)
        loss = yaw.sum() + pitch.sum()
        loss.backward()
        
        # Check at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"
    
    def test_different_batch_sizes(self, model):
        """Test model works with different batch sizes."""
        model.eval()
        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, 1, 64, 64)
            with torch.no_grad():
                yaw, pitch = model(input_tensor)
            assert yaw.shape[0] == batch_size
            assert pitch.shape[0] == batch_size


class TestTemporalPredictor:
    """Test cases for TemporalPredictor."""
    
    @pytest.fixture
    def model(self):
        """Create temporal predictor instance."""
        return TemporalPredictor(input_dim=2, hidden_dim=64, num_layers=2)
    
    @pytest.fixture
    def sample_sequence(self):
        """Create sample sequence."""
        # [batch_size, sequence_length, input_dim]
        return torch.randn(2, 10, 2)
    
    def test_temporal_initialization(self, model):
        """Test temporal model initialization."""
        assert model is not None
    
    def test_temporal_forward(self, model, sample_sequence):
        """Test temporal prediction."""
        model.eval()
        with torch.no_grad():
            predictions = model(sample_sequence)
        
        # Should predict next position
        assert predictions.shape == (2, 2), f"Expected (2, 2), got {predictions.shape}"
    
    def test_sequence_lengths(self, model):
        """Test with different sequence lengths."""
        model.eval()
        for seq_len in [5, 10, 20]:
            input_seq = torch.randn(1, seq_len, 2)
            with torch.no_grad():
                output = model(input_seq)
            assert output.shape == (1, 2)


def test_angular_error_computation():
    """Test angular error metric."""
    # Same direction should give 0 error
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([0.0, 0.0])
    
    # Compute unit vectors
    pred_vec = torch.stack([
        torch.cos(pred[0]) * torch.cos(pred[1]),
        torch.sin(pred[0]) * torch.cos(pred[1]),
        torch.sin(pred[1]),
    ])
    
    target_vec = torch.stack([
        torch.cos(target[0]) * torch.cos(target[1]),
        torch.sin(target[0]) * torch.cos(target[1]),
        torch.sin(target[1]),
    ])
    
    dot_product = torch.dot(pred_vec, target_vec)
    angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    error_deg = angle * 180.0 / np.pi
    
    assert error_deg < 1.0, "Same direction should have near-zero error"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
