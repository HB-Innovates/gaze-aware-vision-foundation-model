"""CNN-based gaze direction predictor."""

import torch
import torch.nn as nn
from typing import Tuple


class GazePredictor(nn.Module):
    """CNN-based model for predicting gaze direction from eye images.
    
    Predicts 3D gaze vectors (yaw, pitch, roll) from eye region images.
    
    Args:
        input_channels: Number of input image channels (default: 3)
        base_channels: Base number of channels (default: 64)
        output_dim: Output dimension (3 for yaw/pitch/roll) (default: 3)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        output_dim: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, eye_images: torch.Tensor) -> torch.Tensor:
        """
        Predict gaze direction from eye images.
        
        Args:
            eye_images: Eye region images [batch_size, channels, height, width]
                       Expected input size: [B, 3, 64, 64]
            
        Returns:
            Gaze vectors [batch_size, 3] (yaw, pitch, roll in radians)
        """
        # Extract features
        features = self.conv_layers(eye_images)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Regress gaze direction
        gaze_vector = self.regressor(pooled)
        
        return gaze_vector
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded gaze predictor from {checkpoint_path}")


class BinocularGazePredictor(nn.Module):
    """Binocular gaze predictor using both left and right eye images.
    
    Args:
        eye_predictor: Single eye gaze predictor
        fusion_method: Method to fuse binocular predictions ('average', 'learned')
    """
    
    def __init__(
        self,
        eye_predictor: GazePredictor,
        fusion_method: str = 'learned',
    ):
        super().__init__()
        
        self.left_eye = eye_predictor
        self.right_eye = eye_predictor
        self.fusion_method = fusion_method
        
        if fusion_method == 'learned':
            self.fusion = nn.Sequential(
                nn.Linear(6, 32),  # 3 from each eye
                nn.ReLU(),
                nn.Linear(32, 3),
            )
    
    def forward(
        self,
        left_eye_images: torch.Tensor,
        right_eye_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict gaze from binocular images.
        
        Args:
            left_eye_images: Left eye images [B, 3, 64, 64]
            right_eye_images: Right eye images [B, 3, 64, 64]
            
        Returns:
            Fused gaze vector [B, 3]
        """
        left_gaze = self.left_eye(left_eye_images)
        right_gaze = self.right_eye(right_eye_images)
        
        if self.fusion_method == 'average':
            return (left_gaze + right_gaze) / 2
        elif self.fusion_method == 'learned':
            combined = torch.cat([left_gaze, right_gaze], dim=1)
            return self.fusion(combined)
