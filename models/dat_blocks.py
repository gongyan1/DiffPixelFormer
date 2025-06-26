# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.runner import auto_fp16
from timm.models.layers import to_2tuple, trunc_normal_

class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        
        super().__init__()

        window_size = to_2tuple(window_size)
        self.nc = dim
        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        # self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        # self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w)) 
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        
        # padding workaround
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[0]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[1]
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        H_ori, W_ori = H, W
        _, _, H, W = x.shape
        
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        
        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C
        
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total) # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
            
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x)) # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H_ori, :W_ori].contiguous()
        
        return x, None, None

class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

    def forward(self, x):
        
        B, C, H, W = x.size()
        
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[0]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[1]
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        H_ori, W_ori = H, W
        
        _, _, H, W = x.size()
        
        img_mask = torch.zeros(H, W, device=x.device)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        attn_mask.masked_fill_(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x = super().forward(shifted_x, attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H_ori, :W_ori].contiguous()

        return x, None, None


class LocalAttention_mm(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.nc = dim
        self.proj_q = nn.Linear(dim, dim)  # Projection for query
        self.proj_kv = nn.Linear(dim, 2 * dim)  # Projection for key and value
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Relative position bias
        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, xa, xb, mask=None):
        B, C, H, W = xa.size()

        # Padding workaround
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[0]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[1]
        xa = F.pad(xa, (pad_l, pad_r, pad_t, pad_b))
        xb = F.pad(xb, (pad_l, pad_r, pad_t, pad_b))
        H_ori, W_ori = H, W
        _, _, H, W = xa.shape

        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        # Rearrange xa (query) and xb (key-value)
        xa_total = einops.rearrange(xa, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1])  # B x Nr x Ws x C
        xb_total = einops.rearrange(xb, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1])  # B x Nr x Ws x C

        xa_total = einops.rearrange(xa_total, 'b m n c -> (b m) n c')  # (B * Nr) x Ws x C
        xb_total = einops.rearrange(xb_total, 'b m n c -> (b m) n c')  # (B * Nr) x Ws x C

        # Project query, key, and value
        q = self.proj_q(xa_total)  # (B * Nr) x Ws x C
        kv = self.proj_kv(xb_total)  # (B * Nr) x Ws x 2C
        k, v = torch.chunk(kv, 2, dim=2)  # (B * Nr) x Ws x C, (B * Nr) x Ws x C

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]  # (B * Nr) x H x Ws x C
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)  # (B * Nr) x H x Ws x Ws

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)  # (B * Nr) x H x Ws x Ws

        if mask is not None:
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')

        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)  # (B * Nr) x H x Ws x C
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')  # (B * Nr) x Ws x C
        x = self.proj_drop(self.proj_out(x))  # (B * Nr) x Ws x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1])  # B x C x H x W

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H_ori, :W_ori].contiguous()

        return x, None, None


class ShiftWindowAttention_mm(LocalAttention_mm):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size=None):
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)
        if fmap_size is not None:
            self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size
        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

    def forward(self, xa, xb):
        B, C, H, W = xa.size()

        # Padding workaround
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[0]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[1]
        xa = F.pad(xa, (pad_l, pad_r, pad_t, pad_b))
        xb = F.pad(xb, (pad_l, pad_r, pad_t, pad_b))
        H_ori, W_ori = H, W

        _, _, H, W = xa.size()

        # Create attention mask for shifted windows
        img_mask = torch.zeros(H, W, device=xa.device)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1

        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0], w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW ww ww
        attn_mask.masked_fill_(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Shift xa and xb
        shifted_xa = torch.roll(xa, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        shifted_xb = torch.roll(xb, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # Apply local attention
        sw_x, _, _ = super().forward(shifted_xa, shifted_xb, attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H_ori, :W_ori].contiguous()

        return x, None, None

def window_partition(x, window_size):
    """将输入特征分块，返回 (num_windows*B, window_size*window_size, C)"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_win_h, num_win_w, win_h, win_w, C)
    windows = x.view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    """将窗口还原为原始特征 (B, C, H, W)"""
    B = int(windows.shape[0] // ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, num_win_h, win_h, num_win_w, win_w)
    x = x.view(B, C, H, W)
    return x


class CrossLocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.dim = dim
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        # Q/K/V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x, y: (B, C, H, W)
        B, C, H, W = x.size()
        window_size = self.window_size

        # padding 到窗口倍数
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        y = F.pad(y, (0, pad_r, 0, pad_b))
        _, _, H_pad, W_pad = x.shape

        # 窗口划分
        x_win = window_partition(x, window_size)  # (B*num_windows, window_tokens, C)
        y_win = window_partition(y, window_size)  # (B*num_windows, window_tokens, C)

        # Q/K/V
        Q_x = self.q_proj(x_win)
        K_y = self.k_proj(y_win)
        V_y = self.v_proj(y_win)

        # 多头reshape
        Bn, N, C = Q_x.shape
        Q_x = Q_x.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)  # (Bn, heads, N, head_dim)
        K_y = K_y.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)
        V_y = V_y.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)

        # attention
        attn = (Q_x @ K_y.transpose(-2, -1)) * self.scale  # (Bn, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_x = (attn @ V_y)  # (Bn, heads, N, head_dim)
        out_x = out_x.transpose(1, 2).reshape(Bn, N, C)
        out_x = self.proj_out(out_x)
        out_x = self.proj_drop(out_x)

        # 还原窗口合并
        out_x = window_reverse(out_x, window_size, H_pad, W_pad, C)
        out_x = out_x[:, :, :H, :W].contiguous()  # 去除padding

        return out_x


def shift_feature(x, shift_size):
    # x: (B, C, H, W)
    if shift_size == 0:
        return x
    return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

def reverse_shift_feature(x, shift_size):
    if shift_size == 0:
        return x
    return torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))

class ShiftedCrossLocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, shift_size=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.heads = heads
        self.dim = dim
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x, y: (B, C, H, W)
        B, C, H, W = x.size()
        window_size = self.window_size
        shift_size = self.shift_size

        # padding 到窗口倍数
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        y = F.pad(y, (0, pad_r, 0, pad_b))
        _, _, H_pad, W_pad = x.shape

        # shift
        x_shift = shift_feature(x, shift_size)
        y_shift = shift_feature(y, shift_size)

        # 窗口划分
        x_win = window_partition(x_shift, window_size)  # (B*num_windows, window_tokens, C)
        y_win = window_partition(y_shift, window_size)  # (B*num_windows, window_tokens, C)

        # Q (from x), K/V (from y)
        Q_x = self.q_proj(x_win)
        K_y = self.k_proj(y_win)
        V_y = self.v_proj(y_win)

        # 多头 reshape
        Bn, N, C = Q_x.shape
        Q_x = Q_x.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)
        K_y = K_y.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)
        V_y = V_y.view(Bn, N, self.heads, C // self.heads).transpose(1, 2)

        # attention
        attn = (Q_x @ K_y.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_x = (attn @ V_y)
        out_x = out_x.transpose(1, 2).reshape(Bn, N, C)
        out_x = self.proj_out(out_x)
        out_x = self.proj_drop(out_x)

        # 还原窗口并逆shift
        out_x = window_reverse(out_x, window_size, H_pad, W_pad, C)
        out_x = reverse_shift_feature(out_x, shift_size)
        out_x = out_x[:, :, :H, :W].contiguous()

        return out_x


class DAttentionDCN(nn.Module):
    
    def __init__(
        self, q_size, ns_per_pt, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop,
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.sample_per_pt = ns_per_pt
        self.nc = n_head_channels * n_heads
        # self.use_pe = use_pe
        # self.linear_att = linear_att

        self.proj_off = nn.Linear(self.nc, self.n_heads * self.sample_per_pt * 2)
        self.proj_att = nn.Linear(self.nc, self.n_heads * self.sample_per_pt)
        
        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        reference_pts = self._get_ref_points() # [-1, 1] H W 2
        self.reference = nn.Parameter(reference_pts, requires_grad=False)
        
        self._init_projections()
    
    def _get_ref_points(self):
        
        pts = torch.empty(self.q_h, self.q_w, 2)
        pts.fill_(0.5)
        for i in range(self.q_h):
            for j in range(self.q_w):
                pts[i,j,0].add_(i).div_(self.q_h - 1)
                pts[i,j,1].add_(j).div_(self.q_w - 1)
        pts.mul_(2).sub_(1)
        return pts
    
    def _init_projections(self):
        
        nn.init.constant_(self.proj_off.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.sample_per_pt, 1)
        for i in range(self.sample_per_pt):
            grid_init[:, i, :] *= i + 1
        self.proj_off.bias.data.copy_(grid_init.view(-1))
        
        nn.init.constant_(self.proj_att.weight.data, 0.)
        nn.init.constant_(self.proj_att.bias.data, 0.)

    def forward(self, x):

        B, C, H, W = x.size()
        
        attn = self.proj_att(einops.rearrange(x, 'b c h w -> b (h w) c')) # B (H W) (h Nk)
        attn = einops.rearrange(attn, 'b l (h n) -> b l h n', h=self.n_heads, n=self.sample_per_pt) # B (H W) h Nk
        attn = self.attn_drop(F.softmax(attn, dim=3))
        
        offsets = self.proj_off(einops.rearrange(x, 'b c h w -> b (h w) c')) # B (H W) (h Nk 2)
        off = einops.rearrange(offsets, 'b l (h n p) -> b l h n p', h=self.n_heads, n=self.sample_per_pt, p=2) # B (H W) h Nk 2
        ref = self.reference.reshape(1, H * W, 1, 1, 2).expand(B, -1, self.n_heads, self.sample_per_pt, -1) # B (H W) h Nk 2
        pos = ref + off
        
        v = self.proj_v(x)
        v_ = einops.rearrange(v, 'b (a d) h w -> (b a) d h w', a=self.n_heads, d=self.n_head_channels) # (B h) hc H W
        
        pos = einops.rearrange(pos, 'b l a n p -> (b a) l n p', a=self.n_heads, n=self.sample_per_pt) # (B h) (H W) Nk 2
        
        v_ = F.grid_sample(input=v_, grid=pos[..., (1, 0)].to(v_.dtype), mode='bilinear', align_corners=True) # (B h) hc (H W) Nk
        
        attn_ = einops.rearrange(attn, 'b l a n -> (b a) l n')[:,None,:,:] # (B h) 1 (H W) Nk
        
        v = attn_.mul(v_).sum(dim=3) # (B h) hc (H W)
        v = einops.rearrange(v, '(b a) d (h w) -> b (a d) h w', a=self.n_heads, d=self.n_head_channels, h=H, w=W) # B C H W
        
        x = self.proj_out(v) # B C H W
        y = self.proj_drop(x)

        return y, None, None


class DeformableAttention_gy(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for query, key, value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Sampling offsets and attention weights
        self.sampling_offsets = nn.Linear(dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * n_levels * n_points)
        
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 2).repeat(1, n_levels, n_points, 1)
        for i in range(n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.constant_(self.attention_weights.weight, 0)

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape (B, N, C)
            k: Key tensor of shape (S, N, C)
            v: Value tensor of shape (S, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = q.shape
        S, _, _ = k.shape
        
        # Project inputs
        q = self.q_proj(q)  # (B, N, C)
        k = self.k_proj(k)  # (S, N, C)
        v = self.v_proj(v)  # (S, N, C)
        
        # Generate sampling offsets and attention weights
        offset = self.sampling_offsets(q).view(
            B, N, self.num_heads, self.n_levels, self.n_points, 2
        )  # (B, N, H, L, P, 2)
        
        weights = self.attention_weights(q).view(
            B, N, self.num_heads, self.n_levels * self.n_points
        )  # (B, N, H, L*P)
        weights = F.softmax(weights, dim=-1).view(
            B, N, self.num_heads, self.n_levels, self.n_points
        )  # (B, N, H, L, P)
        
        # Prepare for sampling
        H = W = int(N ** 0.5)  # Assuming square feature maps
        reference_points = self._get_reference_points(H, W, B, q.device)  # (B, N, 2)
        sampling_locations = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + offset
        sampling_locations = sampling_locations.clamp(0, 1)  # Clamp to [0, 1]
        
        # Reshape key and value for sampling
        k = k.view(S, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)  # (S, H, C/H, H, W)
        v = v.view(S, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)  # (S, H, C/H, H, W)
        
        print("k shape:", k.shape)
        print("sampling_locations shape:", sampling_locations.shape)

        # Sample key and value using bilinear interpolation
        sampled_k = self._sample_features(k, sampling_locations)  # (B, N, H, L, P, C/H)
        sampled_v = self._sample_features(v, sampling_locations)  # (B, N, H, L, P, C/H)
        
        # Apply attention weights
        output = (sampled_v * weights.unsqueeze(-1)).sum(dim=(-2, -3))  # (B, N, H, C/H)
        
        # Reshape output
        output = output.view(B, N, C)
        
        # Apply output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
    
    def _get_reference_points(self, H, W, B, device):
        """Generate reference points for the grid."""
        y_ref, x_ref = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device)
        )
        ref_y = y_ref.reshape(-1)[None]  # (1, H*W)
        ref_x = x_ref.reshape(-1)[None]  # (1, H*W)
        ref = torch.stack((ref_x, ref_y), -1)  # (1, H*W, 2)
        return ref.repeat(B, 1, 1)  # (B, H*W, 2)

    def _sample_features(self, features, sampling_locations):
        """Sample features using bilinear interpolation."""
        S, C, height, width = features.shape
        B, N, num_heads, n_levels, n_points, _ = sampling_locations.shape
        
        # Normalize sampling locations to [-1, 1] for grid_sample
        sampling_locations = sampling_locations * 2 - 1
        
        # Reshape features for batch application
        features = features.view(1, S*C, height, width)
        
        # Initialize output tensor
        sampled_features = torch.zeros(
            B, N, num_heads, n_levels, n_points, C,
            device=features.device
        )
        
        # Sample features for each level and point
        for l in range(n_levels):
            for p in range(n_points):
                # Get sampling locations for this level and point
                loc = sampling_locations[:, :, :, l, p, :].view(B*num_heads, N, 1, 2)
                
                # Sample features
                sampled = F.grid_sample(
                    features.expand(B*num_heads, -1, -1, -1),
                    loc,
                    mode='bilinear', padding_mode='zeros', align_corners=False
                )  # (B*num_heads, S*C, 1, N)
                
                # Reshape sampled features
                sampled = sampled.view(B, num_heads, S, C, N).permute(0, 4, 1, 2, 3)  # (B, N, num_heads, S, C)
                
                # Store sampled features
                sampled_features[:, :, :, l, p, :] = sampled.mean(dim=3)  # Average over S dimension
        
        return sampled_features


class DAttentionBaseline(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, ksize, log_cpb, stage_i
    ):

        super().__init__()
        self.fp16_enabled = False
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        self.log_cpb = log_cpb
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            Hk, Wk = x_sampled.size(2), x_sampled.size(3)
            n_sample = Hk * Wk
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn_bias = F.interpolate(attn_bias, size=(H * W, n_sample), mode='bilinear', align_corners=True)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0) # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement) # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_q_grid(H, W, B, dtype, device)

                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)

                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        
        y = self.proj_drop(self.proj_out(out))
        
        return y, None, None

class PyramidAttention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        
        super().__init__()

        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.proj_ds = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
                LayerNormProxy(dim)
            )
    
    def forward(self, x):
        
        B, C, H, W = x.size()
        Nq = H * W
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ds = self.proj_ds(x)
            kv = self.kv(x_ds)
        else:
            kv = self.kv(x)

        k, v = torch.chunk(kv, 2, dim=1)
        Nk = (H // self.sr_ratio) * (W // self.sr_ratio)
        q = q.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k = k.reshape(B * self.num_heads, self.head_dim, Nk)
        v = v.reshape(B * self.num_heads, self.head_dim, Nk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        x = torch.einsum('b m n, b c n -> b c m', attn, v)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None, None

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class TransformerMLPWithConv_CMT(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            nn.GELU(),
            nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x
    

class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x
    
class LDABaseline(nn.Module):
    
    def __init__(
        self, q_size, ker_size, n_heads,
        n_head_channels, n_groups, use_pe, no_off,
        ksize
    ):
        super().__init__()
        self.q_h, self.q_w = q_size
        self.ker_h, self.ker_w = ker_size
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.no_off = no_off
        self.ksize = ksize
        self.ns_per_q = self.ker_h * self.ker_w
        
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        
        # DWConv + LN + GELU + 1x1Conv
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.n_group_channels, self.n_group_channels,
                kernel_size=ksize, stride=1, padding=ksize // 2,
                groups=self.n_group_channels
            ),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(
                self.n_group_channels, 2 * self.ns_per_q,
                kernel_size=1, stride=1, padding=0, bias=False
            )
        )
        
        if self.use_pe:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=1e-2)
        else:
            self.rpe_table = None
    
    @torch.no_grad()
    def _get_ref_points(self, B, H, W, dtype, device):
        
        q_grid_y, q_grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        pad_size = (self.ker_h // 2, self.ker_w // 2)
        # B(1), 2, Hq, Wq -> B(1), 2 * (Kh * Kw), Hq * Wq
        q_grid = torch.stack((q_grid_y, q_grid_x), 0).unsqueeze(0)
        q_grid = F.pad(q_grid, (pad_size[1], pad_size[1], pad_size[0], pad_size[0]), 'replicate')
        q_grid[:, 1, ...].div_(W - 1.0).mul_(2.0).sub_(1.0)
        q_grid[:, 0, ...].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = F.unfold(
            q_grid,
            kernel_size=(self.ker_h, self.ker_w),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1)
        ).reshape(1, 2 * self.ker_h * self.ker_w, H, W)
        
        return ref.expand(B * self.n_groups, -1, -1, -1)
    
    @torch.no_grad()
    def _get_q_grid(self, B, H, W, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    
    def forward(self, x):
        
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q_off = einops.rearrange(
            q,
            'b (g c) h2 w2 -> (b g) c h2 w2',
            g=self.n_groups, c=self.n_group_channels
        )
        offset = self.conv_offset(q_off).contiguous()
        # offset: B * g, (2 * Kh * Kw), Hq, Wq
        ref = self._get_ref_points(B, H, W, dtype, device)
        if self.no_off:
            offset.zero_()
        # Sum up and clip to [-1, 1]
        pos = (offset + ref).clamp(-1., +1.)
        pos = einops.rearrange(
            pos,
            'b (p h1 w1) h2 w2 -> b (h2 w2) (h1 w1) p',
            b=B * self.n_groups, p=2, h1=self.ker_h, w1=self.ker_w, h2=H, w2=W
        )
        k_sampled = F.grid_sample(
            input=k.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True
        )
        v_sampled = F.grid_sample(
            input=v.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True
        )
        # k, v: B * g, Cg, Hq * Wq, Kh * Kw
        k_sampled = k_sampled.reshape(B, C, H * W, self.ker_h * self.ker_w)
        k_sampled = einops.rearrange(
            k_sampled,
            'b (h d) q k -> b h d q k',
            h=self.n_heads, d=self.n_head_channels
        )
        v_sampled = v_sampled.reshape(B, C, H * W, self.ker_h * self.ker_w)
        v_sampled = einops.rearrange(
            v_sampled,
            'b (h d) q k -> b h d q k',
            h=self.n_heads, d=self.n_head_channels
        )
        # B, h, Ch, Nq (,Nk)
        q = q.reshape(B, self.n_heads, self.n_head_channels, H * W)
        q = q.mul(self.scale)
        attn = torch.einsum('b h d q, b h d q k -> b h q k', q, k_sampled)
        
        if self.use_pe:
            rpe_table = self.rpe_table[None, ...].expand(B, -1, -1, -1) # B, h, 2H-1, 2W-1
            q_grid = self._get_q_grid(B, H, W, dtype, device) # B * g H W 2
            q_grid = q_grid.reshape(B * self.n_groups, H * W, 2)
            displacement = (q_grid.unsqueeze(2) - pos).mul(0.5)
            # disp: B * g, Hq * Wq, Kh * Kw, 2
            attn_bias = F.grid_sample(
                input=rpe_table.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                grid=displacement[..., (1, 0)],
                mode='bilinear', align_corners=True)
            # B * g, G_h, Hq * Wq, Kh * Kw
            attn_bias = attn_bias.reshape(B, self.n_heads, H * W, self.ker_h * self.ker_w)
            attn = attn + attn_bias
            
        attn = F.softmax(attn, dim=3)
        out = torch.einsum('b h q k, b h d q k -> b h d q', attn, v_sampled)
        out = out.reshape(B, C, H, W)
        
        y = self.proj_out(out)
        
        return y, pos, ref