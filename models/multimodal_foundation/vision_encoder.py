"""Vision encoder using CLIP architecture."""

import torch
import torch.nn as nn
from typing import Optional
import timm


class VisionEncoder(nn.Module):
    """CLIP-based vision encoder for extracting visual features.
    
    Uses a Vision Transformer (ViT) backbone pretrained with CLIP.
    
    Args:
        model_name: Name of the vision model (default: 'vit_base_patch16_224')
        embed_dim: Output embedding dimension (default: 512)
        pretrained: Whether to load pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone parameters (default: False)
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Load pretrained vision transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[-1]
        
        # Projection head to target embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            
        Returns:
            Visual embeddings [batch_size, embed_dim]
        """
        # Extract features from backbone
        features = self.backbone(images)  # [B, backbone_dim]
        
        # Project to target dimension
        embeddings = self.projection(features)  # [B, embed_dim]
        
        return embeddings
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
