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
        self.patch_size = ensure_tuple_rep(patch_size, 2)
        self.in_channs = in_channs
        self.embed_dim = embed_dims 
        self.proj = nn.Conv2d(
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
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

class MLP(nn.Module):
    def __init__(
        self, 
        in_features:int, 
        out_features:int,
        hidden_features:int,
        act_layer:nn.Module = nn.GELU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
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
    
    def forward(self, x:torch.Tensor):
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
        window_size: tuple[int, int, int] | tuple[int, int],
        embed_dim:int=768,
        num_heads:int= 8,
        qkv_bias:bool = True,
        attn_drop:float = 0.0,
        proj_drop:float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size #window size: [D, H, W]
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim = -1)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.o_prj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),num_heads
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            # Shape: (3, Wd, Wh, Ww)
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            # Shape: (N, N, 3), N = Wd*Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            # (X, Y, Z) -> Flatten: X + WIDTH * ( Y + DEPTH * Z)
            relative_coords[:, :, 0] *= (2*self.window_size[1] - 1) * (2*self.window_size[2] -1)
            relative_coords[:, :, 1] *= (2*self.window_size[2] - 1)

        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            #Shape (N, N, 2), N = Wd*Wh
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] -1
            # (X, Y) -> Flatten: X + WIDTH * Y
            relative_coords[:, :, 0] += 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x:torch.Tensor, mask):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k ,v = qkv[0], qkv[1], qkv[2]
        # (B, num_heads, N, N)
        attn = (q @ k.transpose(2, 3)) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0 , 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B // nw, nw, self.num_heads, N, N ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn) 
        attn = self.attn_drop(attn).to(v.dtype)
        # (B. num_heads, N, N) -> (B, num_heads, N, head_dim) -> (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.o_prj(out))
        return out
        

        

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
        window_size:tuple[int, int, int] | tuple[int, int],
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
        self.norm1 = norm_layer(embed_dim)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.norm2 = norm_layer(embed_dim)
        self.path_drop = nn.Dropout(p=drop_path)
        self.mlp = MLP(embed_dim, embed_dim, mlp_ratio * embed_dim)
        self.w_msa = WindowAttention(window_size, embed_dim, qkv_bias=qkv_bias, 
        attn_drop=attn_drop,)
        self.sw_msa = ShiftedWindowAttention(embed_dim, window_size=self.window_size,qkv_bias=qkv_bias, attn_drop=attn_drop,)
    
    def forward(self, x:torch.Tensor):
        # First pass 
        residual = x 
        x = self.attn_drop(self.w_msa(self.norm1(x)))
        x = residual + x
        residual = x
        x = self.path_drop(self.mlp(self.norm2(x)))
        #Second pass
        residual = x 
        x = self.attn_drop(self.sw_msa(self.norm1(x)))
        x = residual + x
        residual = x
        x = self.path_drop(self.mlp(self.norm2(x)))
        x = residual + x
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
        img_size: tuple[int, int, int] = (96, 96, 96),
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
        )

    def load_pretrained(self, weights_path: str | Path) -> None:
        """Load pretrained Swin encoder weights (partial weight loading)."""
        self.model.load_from(torch.load(weights_path, weights_only=True))

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
    imgs = torch.rand((4, 1, 96, 96, 96))
    print(patch_embed(rand).shape)
    # swin_unetr = SwinUNETRWrapper()
    # print(swin_unetr(imgs).shape)
    mlp = MLP(768, 768, 768 * 4)
    img = torch.rand(4, 196, 768)
    print(mlp.act)