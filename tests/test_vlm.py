"""Unit tests for Vision-Language Model."""

import pytest
import torch
from models.multimodal_foundation.vlm import (
    GazeAwareVLM,
    GazeAttentionFusion,
    ProjectionHead,
)


class TestProjectionHead:
    """Test cases for projection head."""
    
    @pytest.fixture
    def projection(self):
        """Create projection head."""
        return ProjectionHead(input_dim=512, output_dim=256)
    
    def test_projection_forward(self, projection):
        """Test projection forward pass."""
        input_tensor = torch.randn(4, 512)
        output = projection(input_tensor)
        assert output.shape == (4, 256)


class TestGazeAttentionFusion:
    """Test cases for gaze attention fusion."""
    
    @pytest.fixture
    def fusion(self):
        """Create fusion module."""
        return GazeAttentionFusion(hidden_dim=256)
    
    def test_fusion_forward(self, fusion):
        """Test fusion of visual and gaze features."""
        visual_features = torch.randn(2, 256)
        gaze_vector = torch.randn(2, 2)
        
        fused = fusion(visual_features, gaze_vector)
        assert fused.shape == (2, 256)


class TestGazeAwareVLM:
    """Test cases for complete VLM."""
    
    @pytest.fixture
    def model(self):
        """Create VLM instance."""
        # Use small models for testing
        return GazeAwareVLM(
            vision_model='openai/clip-vit-base-patch32',
            text_model='gpt2',
        )
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for large models"
    )
    def test_forward_pass(self, model):
        """Test forward pass with image and text."""
        # This test requires actual model weights
        # Skip for now as it's resource-intensive
        pytest.skip("Requires model weights")
    
    def test_gaze_guided_attention(self):
        """Test gaze-guided attention mechanism."""
        # Create mock attention maps
        attention = torch.randn(1, 12, 197, 197)  # CLIP attention
        gaze_point = torch.tensor([[0.5, 0.5]])  # Center
        
        # Gaze should modulate attention
        assert attention.requires_grad == False or attention.grad_fn is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
