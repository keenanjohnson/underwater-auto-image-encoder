import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from einops import repeat


from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn



class LearnedPositionalEncoding(nn.Module):
    def __init__(self,seq_length=256, embedding_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x
        #print(seq_length,embedding_dim)

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

#Mamba
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

#Mamba
class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 3, expand = 2, drop_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        #self.mamba = Mamba(
        #        d_model=input_dim, # Model dimension d_model
        #        d_state=d_state,  # SSM state expansion factor
        #        d_conv=d_conv,    # Local convolution width
        #        expand=expand,    # Block expansion factor
        #        dropout=drop_rate,
        #)
        self.mamba = SS2D(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=drop_rate)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]  #batch,channel
        assert C == self.input_dim

        x1 = self.norm1(x.permute(0, 2, 3, 1)) #B,C,H,W -> B.H,W,C
        x1 = self.mamba(x1) + self.skip_scale * x.permute(0, 2, 3, 1)
        x2 = self.norm2(x1)
        out=x2.permute(0, 3, 1, 2) #B,H,W,C -> B,C,H,W
        return out











#频域注意力
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, h//2+1, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        #weight = torch.view_as_complex(self.complex_weight)
        weight = torch.view_as_complex(self.complex_weight.clone().contiguous())
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class GF_Layer(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super(GF_Layer,self).__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class GF_block(nn.Module):
    
    def __init__(self,in_Channel,h,w,dropout_rate=0.0,mlp_ratio=4.,norm_layer=nn.LayerNorm):
    #input B,C,H,W
        super(GF_block,self).__init__()
        self.h=h
        self.w=w
        H_Multi_W=h*w
        self.seq_length=H_Multi_W
        self.embedding_dim=in_Channel
        self.drop_rate=dropout_rate
        self.position_encoding = LearnedPositionalEncoding(self.seq_length,self.embedding_dim)
        self.pe_dropout = nn.Dropout(p=self.drop_rate)

        self.blocks =GF_Layer(dim=self.embedding_dim, mlp_ratio=mlp_ratio,drop=self.drop_rate, drop_path=self.drop_rate, h=h, w=w)
        self.norm = norm_layer(self.embedding_dim)


    def reshape_output(self,x): #将transformer的输出resize为原来的特征图尺寸
        x = x.view(
            x.size(0),
            int(self.h),
            int(self.w),
            self.embedding_dim,
             )#B,16,16,512
        x = x.permute(0, 3, 1, 2).contiguous()

        return x





    def forward(self, x):
        B,c,h,w = x.shape
        #print(x.shape)
        x=x.permute(0, 2, 3, 1).contiguous()  # B,c,h,w->B,h,w,c
        x=x.view(x.size(0), -1, c) #B,16,16,512->B,16*16,512
        #print(x.shape)
        x= self.position_encoding(x)#位置编码
        x= self.pe_dropout(x) #预dropout层
        x = self.blocks(x)
        #print(x.shape)
        x = self.norm(x)
        #print(x.shape)
        x=self.reshape_output(x)



        #x = self.patch_embed(x)
        #x = x + self.pos_embed
        #x = self.pos_drop(x)
        #x = self.blocks(x)

        #x = self.norm(x).mean(1)
        return x






class SF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path, H,W):
        """ FWSA and Mamba_Block
        """
        super(SF_Block, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.drop_path = drop_path
        #self.input_resolution = input_resolution
        self.H=H
        self.W=W




        #self.trans_block = WMSA_Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.Spec_block=GF_block(self.in_channels,self.H,self.W,dropout_rate=self.drop_path)
        self.mamba_block = MambaLayer(input_dim=self.in_channels, output_dim=self.in_channels)




        self.conv1_1 = nn.Conv2d(self.in_channels, self.in_channels*2, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.in_channels*2, self.out_channels, 1, 1, 0, bias=True)



    def forward(self, x):

        spec_x, mamba_x = torch.split(self.conv1_1(x), (self.in_channels, self.in_channels), dim=1)# B,C,H,W
        spec_x=self.Spec_block(spec_x)+spec_x
        mamba_x=self.mamba_block(mamba_x)+mamba_x
        #trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        #print(trans_x.shape)
        #trans_x = self.trans_block(trans_x)
        #trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((spec_x, mamba_x), dim=1))
        x = x + res

        return x

































