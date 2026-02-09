"""Gaze-guided attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GazeGuidedAttention(nn.Module):
    """Multi-head attention guided by gaze information.
    
    Implements attention mechanism where gaze embeddings guide
    the attention to relevant visual regions.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Gaze context projection for attention modulation
        self.gaze_context_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        gaze_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply gaze-guided attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            gaze_context: Gaze context for attention modulation [batch_size, embed_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            Attention weights (optional) [batch_size, num_heads, seq_len, seq_len]
        """
        # Handle 2D input (add sequence dimension)
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, D]
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, L, D/H]
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [B, H, L, L]
        
        # Modulate attention with gaze context if provided
        if gaze_context is not None:
            gaze_bias = self.gaze_context_proj(gaze_context)  # [B, D]
            gaze_bias = gaze_bias.view(batch_size, 1, 1, self.embed_dim)
            gaze_bias = gaze_bias.expand(-1, self.num_heads, seq_len, -1)
            # Add gaze modulation to attention
            attn_weights = attn_weights + gaze_bias.mean(dim=-1, keepdim=True)
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D/H]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Remove sequence dimension if input was 2D
        if is_2d:
            output = output.squeeze(1)  # [B, D]
        
        if return_attention:
            return output, attn_weights
        return output, None


class SpatialGazeAttention(nn.Module):
    """Spatial attention mechanism guided by gaze heatmaps.
    
    Used for foveated rendering and gaze-aware image processing.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channels = channels
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        feature_maps: torch.Tensor,
        gaze_heatmap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply spatial gaze attention to feature maps.
        
        Args:
            feature_maps: Input feature maps [B, C, H, W]
            gaze_heatmap: Gaze heatmap [B, 1, H, W] (optional)
            
        Returns:
            Attended feature maps [B, C, H, W]
        """
        # Channel attention
        channel_weights = self.channel_attention(feature_maps)
        features = feature_maps * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        max_pool, _ = torch.max(features, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_conv(spatial_input)
        
        # Modulate with gaze heatmap if provided
        if gaze_heatmap is not None:
            spatial_weights = spatial_weights * gaze_heatmap
        
        output = features * spatial_weights
        
        return output
