"""Unit tests for model components."""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.multimodal_foundation import (
        GazeAwareVisionModel,
        VisionEncoder,
        GazeEncoder,
        GazeGuidedAttention,
    )
    from models.gaze_tracking import (
        GazePredictor,
        TemporalGazePredictor,
        GazeCalibration,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Models not yet available")


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestVisionEncoder:
    """Test vision encoder."""
    
    def test_forward_pass(self):
        model = VisionEncoder(embed_dim=512)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 512)
    
    def test_freeze_unfreeze(self):
        model = VisionEncoder(freeze_backbone=True)
        # Check backbone is frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad
        
        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestGazeEncoder:
    """Test gaze encoder."""
    
    def test_forward_pass(self):
        model = GazeEncoder(input_dim=3, embed_dim=512)
        x = torch.randn(2, 3)
        output = model(x)
        assert output.shape == (2, 512)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestGazeAwareVisionModel:
    """Test complete multi-modal model."""
    
    def test_forward_with_gaze(self):
        model = GazeAwareVisionModel()
        images = torch.randn(2, 3, 224, 224)
        gaze = torch.randn(2, 3)
        output, _ = model(images, gaze)
        assert output.shape == (2, 1024)
    
    def test_forward_without_gaze(self):
        model = GazeAwareVisionModel()
        images = torch.randn(2, 3, 224, 224)
        output, _ = model(images)
        assert output.shape == (2, 1024)
    
    def test_return_attention(self):
        model = GazeAwareVisionModel()
        images = torch.randn(2, 3, 224, 224)
        gaze = torch.randn(2, 3)
        output, attention = model(images, gaze, return_attention=True)
        assert output.shape == (2, 1024)
        assert attention is not None


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestGazePredictor:
    """Test gaze predictor."""
    
    def test_forward_pass(self):
        model = GazePredictor()
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        assert output.shape == (2, 3)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestTemporalGazePredictor:
    """Test temporal gaze predictor."""
    
    def test_forward_pass(self):
        model = TemporalGazePredictor(prediction_horizon=5)
        x = torch.randn(2, 10, 3)  # batch, seq_len, gaze_dim
        predictions, hidden = model(x)
        assert predictions.shape == (2, 5, 3)
    
    def test_predict_next_frame(self):
        model = TemporalGazePredictor()
        x = torch.randn(2, 10, 3)
        next_frame = model.predict_next_frame(x)
        assert next_frame.shape == (2, 3)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestGazeCalibration:
    """Test gaze calibration."""
    
    def test_forward_pass(self):
        model = GazeCalibration()
        x = torch.randn(2, 3)
        output = model(x)
        assert output.shape == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
