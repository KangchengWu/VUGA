
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm
class GA_conv(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=5 // 2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=7 // 2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=9, padding=9 // 2, groups=dim)
        self.proj = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1)
    def forward(self,x):
        id = x
        conv1 = self.conv1(id)
        conv2 = self.conv2(id)
        conv3 = self.conv3(id)
        
        x = (conv1+conv2+conv3)/3.0+id
        id = x
        
        x = self.proj(x)
        
        return id+x

class GA(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.project1 = nn.Linear(dim,dim//4)
        self.non_linear = F.gelu
        self.project2 = nn.Conv2d(in_channels=dim//4,out_channels=dim,kernel_size=1)
        self.norm = LayerNorm2D(num_channels=dim)
        self.dconv = GA_conv(dim=dim//4)
        self.drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(1, dim, 1, 1))
    def forward(self,x):  # b c h w 
        b,c,h,w = x.shape
        id = x
        x = self.norm(x)*self.gamma+x*self.gammax
        x = rearrange(x, 'b c h w -> b (h w) c')
       # x = self.dcn_v4(x)
        x = self.project1(x)
        x = rearrange(x,'b (h w) c-> b c h w',h=h,w=w)
        x = self.dconv(x)
        x = self.project2(x)
        x = self.drop(x)
        return x+id
