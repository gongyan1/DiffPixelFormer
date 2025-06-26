import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    B = int(windows.shape[0] // ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x

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

# 用法举例：一层普通cross local-attention + 一层shifted cross local-attention
if __name__ == "__main__":
    B, C, H, W = 2, 64, 56, 56
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    swin_cross_layers = nn.Sequential(
        ShiftedCrossLocalAttention(dim=64, heads=4, window_size=7, shift_size=0),
        ShiftedCrossLocalAttention(dim=64, heads=4, window_size=7, shift_size=7//2)
    )
    out = x
    for layer in swin_cross_layers:
        out = layer(out, y)
    print(out.shape)  # (B, C, H, W)