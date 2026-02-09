"""Cross-modal projection for vision-gaze fusion."""

import torch
import torch.nn as nn


class CrossModalProjection(nn.Module):
    """Project and fuse vision and gaze embeddings.
    
    Combines vision and gaze modalities through learned projections
    and fusion mechanisms.
    
    Args:
        vision_dim: Dimension of vision embeddings
        gaze_dim: Dimension of gaze embeddings
        output_dim: Dimension of fused output embeddings
        fusion_method: Fusion method ('concat', 'add', 'multiply') (default: 'concat')
    """
    
    def __init__(
        self,
        vision_dim: int,
        gaze_dim: int,
        output_dim: int,
        fusion_method: str = 'concat',
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.gaze_dim = gaze_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
        # Gaze projection
        self.gaze_proj = nn.Sequential(
            nn.Linear(gaze_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        elif fusion_method in ['add', 'multiply']:
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        vision_embeds: torch.Tensor,
        gaze_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse vision and gaze embeddings.
        
        Args:
            vision_embeds: Vision embeddings [batch_size, vision_dim]
            gaze_embeds: Gaze embeddings [batch_size, gaze_dim]
            
        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        # Project both modalities
        vision_proj = self.vision_proj(vision_embeds)
        gaze_proj = self.gaze_proj(gaze_embeds)
        
        # Fuse modalities
        if self.fusion_method == 'concat':
            fused = torch.cat([vision_proj, gaze_proj], dim=-1)
            output = self.fusion(fused)
        elif self.fusion_method == 'add':
            output = vision_proj + gaze_proj
        elif self.fusion_method == 'multiply':
            output = vision_proj * gaze_proj
        
        return output
