import torch
import torch.nn as nn
from ops_dcnv3.modules.dcnv3 import DCNv3_pytorch


class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.dcn = DCNv3_pytorch(channels=d_model,kernel_size=kernel_size,dw_kernel_size=dw_kernel_size,pad=pad,stride=stride,dilation=dilation,group=group)
        
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self,x):
        u = x.clone()
        x = self.conv0(x)
        attn = x.permute(0,2,3,1)
        attn = self.dcn(attn)
        attn = attn.permute(0,3,1,2)
        attn = self.conv(attn)
        return u*attn
    

class SDA(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    