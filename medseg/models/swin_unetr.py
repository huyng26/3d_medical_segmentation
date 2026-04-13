"""Swin UNETR wrapper around ``monai.networks.nets.SwinUNETR`` (Phase 3).

Fine-tunes the pretrained Swin Transformer encoder with a CNN decoder.
Optionally loads publicly available pretrained weights at construction time.
"""
from __future__ import annotations
from pathlib import Path
from turtle import forward
from monai.networks.blocks.mlp import MLPBlock
import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT 
from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn.functional as F

def window_partition(x, window_size):
    b, d, h, w, c = x.shape
    x = x.view(
        b, 
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size:int, 
        patch_size:int, 
        in_channs: int,
        embed_dims: int,
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.patch_size = patch_size 
        self.in_channs = in_channs
        self.embed_dim = embed_dims 
        self.patch_embed = nn.Conv2d(
            in_channels=in_channs,
            out_channels=embed_dims, 
            kernel_size=patch_size,
            stride = patch_size,
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.position_ids = torch.arange(self.num_patches).expand(1, -1)
        self.pos_embed = nn.Embedding(self.num_patches, self.embed_dim)
        self.norm = norm_layer(embed_dims)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed(self.position_ids)
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(
        self, 
        in_features:int, 
        out_features:int,
        hidden_features:int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim:int,
        num_heads:int,
        qkv_bias:bool = True,
        attn_drop:float = 0.0,
        proj_drop:float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim 
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.o_prj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.o_prj(out)
        out = self.proj_drop(out)
        return out

class WindowAttention(nn.Module):
    def __init__(
        self,
        embed_dim:int,
        num_heads:int,
        window_size:tuple[int, int],
        qkv_bias:bool = True,
        attn_drop:float = 0.0,
        proj_drop:float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
    
    def forward(self, x:torch.Tensor):
        pass 

class ShiftedWindowAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        pass 

class SwinTransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim:int, 
        num_heads:int, 
        window_size:tuple[int, int], 
        mlp_ratio:int = 4,
        qkv_bias:bool = True, 
        drop:float = 0.0, 
        attn_drop:float = 0.0, 
        drop_path:float = 0.0,
        act_layer:nn.Module = nn.GELU,
        norm_layer:nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm = norm_layer(embed_dim)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.path_drop = nn.Dropout(p=drop_path)
        self.mlp = MLP(embed_dim, embed_dim, mlp_ratio * embed_dim)
        self.w_msa = WindowAttention(embed_dim, window_size=self.window_size,qkv_bias=qkv_bias, 
        attn_drop=attn_drop,)
        self.sw_msa = ShiftedWindowAttention(embed_dim, window_size=self.window_size,qkv_bias=qkv_bias, attn_drop=attn_drop,)
    
    def forward(self, x:torch.Tensor):
        # First pass 
        residual = x 
        x = self.w_msa(self.norm(x))
        x = residual + x
        residual = x
        x = self.mlp(self.norm(x))
        #Second pass
        residual = x 
        x = self.sw_msa(self.norm(x))
        x = residual + x
        residual = x
        x = self.mlp(self.norm(x))
        return x

class SwinUNETRWrapper(nn.Module):
    """Thin wrapper that constructs MONAI's SwinUNETR and optionally loads weights.

    Args:
        in_channels:      Number of input channels (1 for CT).
        out_channels:     Number of segmentation classes.
        img_size:         Patch spatial size the model expects (must match training patches).
        feature_size:     Swin embedding dimension (MONAI default 48).
        use_checkpoint:   Enable gradient checkpointing to reduce VRAM usage.
        weights_path:     Path or URL to pretrained weights; ``None`` to skip.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        img_size: tuple[int, int, int] = (128, 128, 64),
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            weights_path=weights_path,
        )

    def load_pretrained(self, weights_path: str | Path) -> None:
        """Load pretrained Swin encoder weights (partial weight loading)."""
        self.model.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    patch_embed = PatchEmbed(
        patch_size=16,
        in_channs=3,
        embed_dims=768,
        image_size=224
    )
    rand = torch.rand((4, 3, 224, 224))
    print(patch_embed(rand).shape)