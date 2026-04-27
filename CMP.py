#timm的基本使用
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torchvision import transforms
from thop import profile
from MyDataset import MyDataset
#from my_dataset import MyDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math, copy
torch.cuda.empty_cache()
from torch import nn
from torchvision import models
from PIL import Image
from ops_dcnv3.modules.dcnv3 import DCNv3_pytorch
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from PaperCode.Newmode_with_crop.GA import MONA
from PaperCode.Newmode_with_crop.SDA import DSA
import os
class CMP(nn.Module):
    def __init__(self, dim, out_dim, num_heads, bias):
        super().__init__()
        self.DCN = DCNv3_pytorch(channels=dim)
        self.Layer_norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)
        self.mona = MONA(dim=dim)
        self.dsa = DSA(d_model=dim,kernel_size=3,dw_kernel_size=5,pad=1,stride=1,dilation=1,group=1)
    def forward(self,x):
        residual = rearrange(x,'b w h c-> b c h w')
        id = self.gelu(self.Layer_norm(self.DCN(x))) #[b w h c]
        id = rearrange(id,'b w h c-> b c h w')
        b,c,h,w = id.shape
        #global
        qkv = self.qkv_dwconv(self.qkv(id))
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        global_out = self.project_out(out)+self.project_out(residual)
       # global_out = rearrange(global_out,'b c h w-> b h w c')
        #local
        # local_out = self.mona(id)
        # local_out = self.dsa(id)
        # local_out = self.project_out(local_out)    
        #return global_out+local_out
        return global_out
        #return local_out