import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce, lru_cache
from operator import mul
import numpy as np

class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".
    Args:
        x: (N, D, H, W, C)
    Returns:
        x: (N, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

@lru_cache()
def compute_mask(N, D, H, W, window_patch_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((N, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for n in slice(-window_patch_size[0]), slice(-window_patch_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for d in slice(-window_patch_size[1]), slice(-window_patch_size[1], -shift_size[0]), slice(-shift_size[1], None):
            for h in slice(-window_patch_size[2]), slice(-window_patch_size[2], -shift_size[1]), slice(-shift_size[2], None):
                for w in slice(-window_patch_size[3]), slice(-window_patch_size[3], -shift_size[2]), slice(-shift_size[3], None):
                    img_mask[n, d, h, w, :] = cnt
                    cnt += 1
    #print(img_mask.squeeze(-1))
    mask_windows = window_partition(img_mask, window_patch_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    #print('MASK WINDOW', mask_windows)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    #print('ATTN MASK', attn_mask)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

def window_partition(x, window_patch_size):
    """ Partition the input into windows. Attention will be conducted within the windows.
    Args:
        x: (N, D, H, W, C) #meshblocks x (nx1 x nx2 x nx3) x #channels
        window_patch_size: (Nw, Dw, Hw, Ww) window_patch_size x (patch_dimx1 x patch_dimx2 x patch_dimx3)
    Returns:
        windows: (N*num_windows, window_patch_size*window_patch_size, C)
    """
    N, D, H, W, C = x.shape
    x = x.view(N // window_patch_size[0], window_patch_size[0], D // window_patch_size[1], window_patch_size[1], H // window_patch_size[2],
               window_patch_size[2], W // window_patch_size[3], window_patch_size[3], C)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7, 8).contiguous().view(-1, reduce(mul, window_patch_size), C)

    return windows

def window_reverse(windows, window_patch_size, N, D, H, W):
    x = windows.view(N // window_patch_size[0], D // window_patch_size[1], H // window_patch_size[2], W // window_patch_size[3], window_patch_size[0], window_patch_size[1],
                     window_patch_size[2], window_patch_size[3], -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7, 8).contiguous().view(N, D, H, W, -1)
    return x

def reshape_3d(x, grid_size,):
    return x.reshape((-1,)+grid_size)

def roll_block(x, shift_size, reverse=False):
    '''
    meshblock: torch.Tensor size (N, D, H, W, C) #meshblocks x (nx1 x nx2 x nx3) x #channels
    shift_size: tuple (St, Sd, S)
    '''
    if reverse:
        shift_size = [-s for s in list(shift_size)]
    return torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(0,1,2,3))

class WindowAttention(nn.Module):
    """ Window based multi-head mutual attention and self attention.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, mut_attn=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mut_attn = mut_attn

        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # mutual attention with sine position encoding
        if self.mut_attn:
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)

        # mutual attention
        if self.mut_attn:
            qkv = self.qkv_mut(x).reshape(B_, N, 3, self.num_heads,
                                                                               C // self.num_heads).permute(2, 0, 3, 1, 4)
            (q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), torch.chunk(qkv[1], 2, dim=2), torch.chunk(
                qkv[2], 2, dim=2)  # B_, nH, N/2, C
            x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
            x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x


class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_patch_size=(6, 8, 8),
                 shift_size=(0, 0, 0),
                 mut_attn=True,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_patch_size = window_patch_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn
        assert 0 <= self.shift_size[0] < self.window_patch_size[0], "shift_size must in 0-window_patch_size"
        assert 0 <= self.shift_size[1] < self.window_patch_size[1], "shift_size must in 0-window_patch_size"
        assert 0 <= self.shift_size[2] < self.window_patch_size[2], "shift_size must in 0-window_patch_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        N, D, H, W, C = x.shape
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_tf = pad_l = pad_t = pad_d0 = 0
        pad_tb = (self.window_patch_size[0] - N % self.window_patch_size[0]) % self.window_patch_size[0]
        pad_d1 = (self.window_patch_size[1] - D % self.window_patch_size[1]) % self.window_patch_size[1]
        pad_b = (self.window_patch_size[2] - H % self.window_patch_size[2]) % self.window_patch_size[2]
        pad_r = (self.window_patch_size[3] - W % self.window_patch_size[3]) % self.window_patch_size[3]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1, pad_tf, pad_tb), mode='constant')
        #print(f'x after padding: {x.shape}')
        Np, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = roll_block(x, self.shift_size, reverse=False)
            attn_mask = mask_matrix
            #print(f'x after rolling: {shifted_x.shape}')

        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_patch_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_patch_size, Np, Dp, Hp, Wp)  # N' D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = roll_block(x, self.shift_size, reverse=True)
        else:
            x = shifted_x

        if pad_tb>0 or pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:N, :D, :H, :W, :]

        x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (N, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x
    
class TMSAG(nn.Module):
    """ Temporal Mutual Self Attention Group (TMSAG).
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_patch_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 dim_out,
                 depth,
                 num_heads,
                 window_patch_size=[2, 6, 8, 8],
                 shift_size=None,
                 mut_attn=True,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.window_patch_size = window_patch_size
        self.shift_size = list(i // 2 for i in window_patch_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            TMSA(
                dim=dim,
                num_heads=num_heads,
                window_patch_size=window_patch_size,
                shift_size=[0, 0, 0, 0] if i % 2 == 0 else self.shift_size,
                mut_attn=mut_attn,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])
        self.linear_out = nn.Linear(dim,dim_out)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        N, D, H, W, C = x.shape
        Np = int(np.ceil(N / self.window_patch_size[0])) * self.window_patch_size[0]
        Dp = int(np.ceil(D / self.window_patch_size[1])) * self.window_patch_size[1]
        Hp = int(np.ceil(H / self.window_patch_size[2])) * self.window_patch_size[2]
        Wp = int(np.ceil(W / self.window_patch_size[3])) * self.window_patch_size[3]
        attn_mask = compute_mask(Np, Dp, Hp, Wp, self.window_patch_size, self.shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(N, D, H, W, -1)
        #x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.linear_out(x)

        return x