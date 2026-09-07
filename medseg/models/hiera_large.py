"""
Hiera-Large Hierarchical Vision Transformer Backbone for MedSAM-2.
Implements multi-stage hierarchical representation, non-intrusive attention extraction,
and guides deep layer selection (Stage 3 / Stage 4) for semantically rich XAI heatmaps.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with explicit attention matrix caching
    for TMME (Transformer Multimodal Explainability) and Attention Rollout.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Cache for explainability hook extraction (M2 handoff)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # [B, num_heads, N, N]
        
        # Retain for explainability hooks without breaking graph
        self.last_attn_weights = attn

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        if return_attn:
            return out, attn
        return out


class MLP(nn.Module):
    """Multilayer Perceptron with GELU activation."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class HieraBlock(nn.Module):
    """
    Individual Transformer block within Hiera-Large hierarchy.
    """
    def __init__(self, dim: int, num_heads: int, stage_idx: int, block_idx: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.stage_idx = stage_idx
        self.block_idx = block_idx
        self.layer_tag = f"stage_{stage_idx}.block_{block_idx}"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class HieraLargeEncoder(nn.Module):
    """
    Hierarchical Vision Transformer (Hiera-Large) Image Encoder.
    4 Stages with depths: [2, 6, 36, 4] and dimensions: [144, 288, 576, 1152].
    """
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple[int, ...] = (144, 288, 576, 1152),
        depths: Tuple[int, ...] = (2, 6, 36, 4),
        num_heads: Tuple[int, ...] = (2, 4, 8, 16),
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.patch_size = patch_size

        # Initial patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dims[0], kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dims[0])

        # Multi-stage blocks
        self.stages = nn.ModuleList()
        self.stage_transitions = nn.ModuleList()

        for s_idx, (dim, depth, heads) in enumerate(zip(embed_dims, depths, num_heads), start=1):
            stage_blocks = nn.ModuleList([
                HieraBlock(dim=dim, num_heads=heads, stage_idx=s_idx, block_idx=b_idx)
                for b_idx in range(depth)
            ])
            self.stages.append(stage_blocks)

            # Downsample transition between stages (except after last stage)
            if s_idx < len(embed_dims):
                next_dim = embed_dims[s_idx]
                transition = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, next_dim)
                )
                self.stage_transitions.append(transition)

        # Neck down-projections for FPN-like multiscale pyramid output
        self.fpn_neck = nn.ModuleList([
            nn.Conv2d(embed_dims[0], 256, kernel_size=1),
            nn.Conv2d(embed_dims[1], 256, kernel_size=1),
            nn.Conv2d(embed_dims[2], 256, kernel_size=1),
            nn.Conv2d(embed_dims[3], 256, kernel_size=1)
        ])

    def get_deep_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Guides layer selection for Member 2 (M2) explainability hooks.
        Identifies and isolates semantically rich deep attention blocks from
        Stage 3 (penultimate high-capacity stage) and Stage 4 (abstract semantic stage).
        These avoid superficial edge-detector artifacts common in early layers.
        """
        deep_layers = []
        # Stage 3 deep blocks (last 4 blocks of stage 3: block_32 to block_35)
        stage_3 = self.stages[2]
        for idx in range(max(0, len(stage_3) - 4), len(stage_3)):
            deep_layers.append((f"stage_3.block_{idx}.attn", stage_3[idx].attn))

        # Stage 4 all blocks (highest semantic abstraction)
        stage_4 = self.stages[3]
        for idx, blk in enumerate(stage_4):
            deep_layers.append((f"stage_4.block_{idx}.attn", blk.attn))

        return deep_layers

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass converting (B, C, H, W) into multiscale pyramid features.
        Also returns dictionary of deep attention maps for XAI hooks.
        """
        B, C, H, W = x.shape
        x_emb = self.patch_embed(x)  # [B, C_0, H/P, W/P]
        _, _, H_p, W_p = x_emb.shape
        
        # Flatten spatial tokens to [B, N, C]
        tokens = x_emb.flatten(2).transpose(1, 2)
        tokens = self.patch_norm(tokens)

        pyramid_features = []
        attention_maps = {}

        curr_h, curr_w = H_p, W_p

        for s_idx, stage_blocks in enumerate(self.stages):
            for b_idx, block in enumerate(stage_blocks):
                # If block is in deep stages, extract attention map
                if s_idx >= 2 and b_idx >= len(stage_blocks) - 2:
                    tokens, attn = block(tokens, return_attn=True)
                    attention_maps[block.layer_tag] = attn
                else:
                    tokens = block(tokens, return_attn=False)

            # Spatial reshaping for pyramid feature
            feat_map = tokens.transpose(1, 2).reshape(B, self.embed_dims[s_idx], curr_h, curr_w)
            proj_feat = self.fpn_neck[s_idx](feat_map)
            pyramid_features.append(proj_feat)

            # Downsample token count if transitioning to next stage
            if s_idx < len(self.stage_transitions):
                tokens = self.stage_transitions[s_idx](tokens)
                # Spatial pooling emulation: halve token resolution if divisible
                if curr_h > 1 and curr_w > 1 and curr_h % 2 == 0 and curr_w % 2 == 0:
                    t_reshaped = tokens.transpose(1, 2).reshape(B, -1, curr_h, curr_w)
                    t_pooled = F.adaptive_avg_pool2d(t_reshaped, (curr_h // 2, curr_w // 2))
                    curr_h, curr_w = curr_h // 2, curr_w // 2
                    tokens = t_pooled.flatten(2).transpose(1, 2)

        return pyramid_features, attention_maps
