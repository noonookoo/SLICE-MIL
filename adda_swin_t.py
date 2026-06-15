from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ADDASwinTinyEncoder", "AnatomyDrivenDualQueryAttention"]


def _to_2d_mask(mask: Optional[torch.Tensor], image_size: Tuple[int, int], device) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.dim() == 4:
        mask = mask.max(dim=1)[0]
    elif mask.dim() != 3:
        raise ValueError(f"mask must be [B, C, H, W] or [B, H, W], got {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) != tuple(image_size):
        mask = F.interpolate(mask.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False).squeeze(1)
    return mask.clamp(0.0, 1.0)


def _pool_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(mask.unsqueeze(1), output_size=size).squeeze(1).clamp(0.0, 1.0)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / ((h // window_size) * (w // window_size)))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * mask.floor()


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        image_size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size)
        patch_size = (int(patch_size), int(patch_size)) if isinstance(patch_size, int) else tuple(patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(int(in_chans), int(embed_dim), kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), (h, w)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(int(dim), int(hidden_dim))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), int(dim))
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

        size = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, self.num_heads))
        self.register_buffer("relative_position_index", self._make_relative_position_index(self.window_size), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _make_relative_position_index(window_size: int) -> torch.Tensor:
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        return relative_coords.sum(-1)

    def _relative_bias(self, n: int) -> torch.Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        return bias.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._relative_bias(n).unsqueeze(0)
        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=x.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x)), attn


class AnatomyDrivenDualQueryAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.query_region_a = nn.Linear(self.dim, self.dim, bias=bool(qkv_bias))
        self.query_region_b = nn.Linear(self.dim, self.dim, bias=bool(qkv_bias))
        self.key = nn.Linear(self.dim, self.dim, bias=bool(qkv_bias))
        self.value = nn.Linear(self.dim, self.dim, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

        size = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, self.num_heads))
        self.register_buffer("relative_position_index", WindowSelfAttention._make_relative_position_index(self.window_size), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _relative_bias(self, n: int) -> torch.Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        return bias.permute(2, 0, 1).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        m_in: torch.Tensor,
        m_ex: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        m_in = m_in.to(device=x.device, dtype=x.dtype).view(b, n, 1).clamp(0.0, 1.0)
        m_ex = m_ex.to(device=x.device, dtype=x.dtype).view(b, n, 1).clamp(0.0, 1.0)

        q_region_a = self.query_region_a(x)
        q_region_b = self.query_region_b(x)
        q = q_region_a * m_ex + q_region_b * m_in
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._relative_bias(n).unsqueeze(0)
        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=x.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x)), attn


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_adda: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.resolution = tuple(resolution)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if min(self.resolution) <= self.window_size:
            self.window_size = min(self.resolution)
            self.shift_size = 0
        self.use_adda = bool(use_adda)
        self.norm1 = nn.LayerNorm(self.dim)
        attn_cls = AnatomyDrivenDualQueryAttention if self.use_adda else WindowSelfAttention
        self.attn = attn_cls(self.dim, self.window_size, int(num_heads), qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(self.dim, int(self.dim * float(mlp_ratio)), dropout=drop)
        self.register_buffer("attn_mask", self._make_attn_mask(), persistent=False)

    def _make_attn_mask(self) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        h, w = self.resolution
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(
        self,
        x: torch.Tensor,
        m_in: Optional[torch.Tensor] = None,
        m_ex: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, w = self.resolution
        b, l, c = x.shape
        if l != h * w:
            raise ValueError(f"expected token length {h*w}, got {l}")
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if m_in is not None:
                m_in = torch.roll(m_in, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if m_ex is not None:
                m_ex = torch.roll(m_ex, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = _window_partition(x, self.window_size)
        attn_map = None
        if self.use_adda:
            if m_in is None or m_ex is None:
                raise ValueError("ADDA stage requires both intra-hepatic and extra-hepatic masks")
            m_in_windows = _window_partition(m_in.unsqueeze(-1), self.window_size).squeeze(-1)
            m_ex_windows = _window_partition(m_ex.unsqueeze(-1), self.window_size).squeeze(-1)
            attn_windows, attn_map = self.attn(x_windows, m_in_windows, m_ex_windows, self.attn_mask)
        else:
            attn_windows, attn_map = self.attn(x_windows, self.attn_mask)

        x = _window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_map


class PatchMerging(nn.Module):
    def __init__(self, resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.resolution = tuple(resolution)
        self.dim = int(dim)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = self.resolution
        b, l, c = x.shape
        if l != h * w:
            raise ValueError(f"expected token length {h*w}, got {l}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("PatchMerging requires even spatial resolution")
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, -1, 4 * c)
        return self.reduction(self.norm(x)), (h // 2, w // 2)


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: List[float],
        downsample: bool,
        use_adda: bool,
    ):
        super().__init__()
        self.resolution = tuple(resolution)
        self.use_adda = bool(use_adda)
        self.blocks = nn.ModuleList()
        for i in range(int(depth)):
            shift = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(
                SwinBlock(
                    dim=dim,
                    resolution=resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    use_adda=use_adda,
                )
            )
        self.downsample = PatchMerging(resolution, dim) if downsample else None

    def forward(
        self,
        x: torch.Tensor,
        m_in: Optional[torch.Tensor] = None,
        m_ex: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            x, attn = block(x, m_in=m_in, m_ex=m_ex)
            if self.use_adda:
                aux[f"adda_attn_block{i}"] = attn
        resolution = self.resolution
        if self.downsample is not None:
            x, resolution = self.downsample(x)
        return x, resolution, aux


class ADDASwinTinyEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.image_size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size)
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(float(drop_rate))
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))

        total_depth = sum(int(d) for d in depths)
        dpr = torch.linspace(0, float(drop_path_rate), total_depth).tolist()
        resolution = self.patch_embed.grid_size
        dim = int(embed_dim)
        self.stages = nn.ModuleList()
        cursor = 0
        for stage_idx, depth in enumerate(depths):
            is_last = stage_idx == len(depths) - 1
            stage = SwinStage(
                dim=dim,
                resolution=resolution,
                depth=int(depth),
                num_heads=int(num_heads[stage_idx]),
                window_size=int(window_size),
                mlp_ratio=float(mlp_ratio),
                qkv_bias=bool(qkv_bias),
                drop=float(drop_rate),
                attn_drop=float(attn_drop_rate),
                drop_path=dpr[cursor : cursor + int(depth)],
                downsample=not is_last,
                use_adda=is_last,
            )
            self.stages.append(stage)
            cursor += int(depth)
            if not is_last:
                resolution = (resolution[0] // 2, resolution[1] // 2)
                dim *= 2
        self.norm = nn.LayerNorm(self.num_features)

    def forward_tokens(
        self,
        x: torch.Tensor,
        m_in: torch.Tensor,
        m_ex: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"x must be [B, C, H, W], got {tuple(x.shape)}")
        _, _, h_img, w_img = x.shape
        image_size = (h_img, w_img)
        m_in_2d = _to_2d_mask(m_in, image_size=image_size, device=x.device)
        m_ex_2d = _to_2d_mask(m_ex, image_size=image_size, device=x.device)
        if m_ex_2d is None:
            m_ex_2d = (1.0 - m_in_2d).clamp(0.0, 1.0)

        x, resolution = self.patch_embed(x)
        x = self.pos_drop(x)
        aux: Dict[str, torch.Tensor] = {}
        for stage_idx, stage in enumerate(self.stages):
            if stage.use_adda:
                m_in_stage = _pool_mask(m_in_2d, stage.resolution)
                m_ex_stage = _pool_mask(m_ex_2d, stage.resolution)
                x, resolution, stage_aux = stage(x, m_in=m_in_stage, m_ex=m_ex_stage)
                aux["m_in_stage4"] = m_in_stage
                aux["m_ex_stage4"] = m_ex_stage
                aux.update(stage_aux)
            else:
                x, resolution, _ = stage(x)
        return self.norm(x), aux

    def forward(
        self,
        x: torch.Tensor,
        m_in: torch.Tensor,
        m_ex: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tokens, aux = self.forward_tokens(x, m_in=m_in, m_ex=m_ex)
        return tokens.mean(dim=1), aux
