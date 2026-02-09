"""User-specific gaze calibration and personalization."""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class GazeCalibration(nn.Module):
    """Personalization module for user-specific gaze calibration.
    
    Adapts gaze predictions to individual user characteristics through
    calibration and fine-tuning.
    
    Args:
        input_dim: Dimension of gaze vector (default: 3)
        calibration_points: Number of calibration points (default: 9)
        adaptation_dim: Dimension of adaptation embeddings (default: 64)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        calibration_points: int = 9,
        adaptation_dim: int = 64,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.calibration_points = calibration_points
        self.adaptation_dim = adaptation_dim
        
        # User-specific adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(input_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, input_dim),
        )
        
        # Calibration bias (learned per user)
        self.calibration_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Calibration scale (learned per user)
        self.calibration_scale = nn.Parameter(torch.ones(input_dim))
    
    def forward(self, gaze_predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply user-specific calibration to gaze predictions.
        
        Args:
            gaze_predictions: Raw gaze predictions [batch_size, 3]
            
        Returns:
            Calibrated gaze predictions [batch_size, 3]
        """
        # Apply learned transformation
        adapted = self.adaptation_net(gaze_predictions)
        
        # Apply calibration scale and bias
        calibrated = adapted * self.calibration_scale + self.calibration_bias
        
        return calibrated
    
    def calibrate(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        num_iterations: int = 100,
        lr: float = 0.01,
    ):
        """
        Calibrate model using user-specific calibration data.
        
        Args:
            predictions: List of predicted gaze vectors
            ground_truth: List of ground truth gaze vectors
            num_iterations: Number of calibration iterations
            lr: Learning rate for calibration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        pred_tensor = torch.stack(predictions)
        gt_tensor = torch.stack(ground_truth)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            calibrated = self.forward(pred_tensor)
            
            # Compute loss
            loss = criterion(calibrated, gt_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (iteration + 1) % 20 == 0:
                print(f"Calibration iteration {iteration + 1}/{num_iterations}, "
                      f"Loss: {loss.item():.6f}")


class MultiUserCalibration:
    """Manage calibration for multiple users.
    
    Stores and loads user-specific calibration models.
    """
    
    def __init__(self):
        self.calibration_models = {}
    
    def add_user(self, user_id: str, calibration_model: GazeCalibration):
        """Add a user-specific calibration model."""
        self.calibration_models[user_id] = calibration_model
    
    def get_user_model(self, user_id: str) -> GazeCalibration:
        """Get calibration model for a specific user."""
        if user_id not in self.calibration_models:
            # Create new calibration model for new user
            self.calibration_models[user_id] = GazeCalibration()
        return self.calibration_models[user_id]
    
    def save_user_calibration(self, user_id: str, path: str):
        """Save user-specific calibration."""
        if user_id in self.calibration_models:
            torch.save(
                self.calibration_models[user_id].state_dict(),
                f"{path}/calibration_{user_id}.pth"
            )
    
    def load_user_calibration(self, user_id: str, path: str):
        """Load user-specific calibration."""
        model = GazeCalibration()
        model.load_state_dict(
            torch.load(f"{path}/calibration_{user_id}.pth", map_location='cpu')
        )
        self.calibration_models[user_id] = model
