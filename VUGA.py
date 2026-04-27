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
from LGFA import DA
from MONA import MONA
from DSA import DSA
backbone = nn.Sequential(*list(models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).children())[:-1])

class VUGA(nn.Module):
    def __init__(self):
        super(VUGA,self).__init__()
        self.layer_1 = backbone[0][0:3]
        self.layer_2 = backbone[0][3:5]
        self.layer_3 = backbone[0][5:7]
        self.layer_4 = backbone[0][7]
        self.layer_finsh = backbone[1:]
        self.backbone = backbone
        self.fc1 = nn.Linear(1280, 512)
        self.bn = nn.BatchNorm2d(768)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.da_192 = DA(dim=192,out_dim=512,num_heads=4,bias=True)
        self.da_384 = DA(dim=384,out_dim=512,num_heads=4,bias=True)
        self.da_768 = DA(dim=768,out_dim=512,num_heads=4,bias=True)
        self.dsa = DSA(d_model=512,kernel_size=3,dw_kernel_size=5,pad=1,stride=1,dilation=1,group=1)
        
        
        self.freezen([self.layer_1,self.layer_2])
        self.adp = nn.AdaptiveAvgPool2d((1,1))
        self.lfa = LFA(in_channels=768,out_channels=768)
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
    def freezen(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False         
    def feature_forward(self,x):
        
        
        layer1_out = self.layer_1(x)  #[28,28,192]
        layer2_out = self.layer_2(layer1_out)#[14,14,384]
        layer3_out = self.layer_3(layer2_out)#[7,7,768]
        layer4_out = self.layer_4(layer3_out)#[7,7,768]
        
        layer1_DA =self.da_192(layer1_out)
        layer2_DA = self.da_384(layer2_out)
        layer3_DA = self.da_768(layer3_out)
        layer4_DA = self.da_768(layer4_out)
        
        layer_3_4 = self.dsa(layer3_DA+layer4_DA)
        
        layer_2_3_4 = self.dsa(layer2_DA+F.interpolate(layer_3_4,scale_factor=2,mode='bilinear',align_corners=True))
        
        layer_1_2_3_4 =self.dsa(layer1_DA+F.interpolate(layer_2_3_4,scale_factor=2,mode='bilinear',align_corners=True))
        layer4_lfa = self.layer_finsh(self.lfa(layer4_out))
        fuse_vector = torch.cat([layer4_lfa,self.adp(layer_1_2_3_4).flatten(1)],dim=1)
        return fuse_vector
    
    def forward(self,x):
        feature = self.feature_forward(x)
        out = self.fc2(self.relu(self.fc1(feature))).squeeze(1)
        return out
    
class LFA(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(LFA, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.dw_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.pointwise_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.dw_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels)
        self.pointwise_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(out_channels)
         
    def forward(self, x):
        x = rearrange(x,'b w h c -> b c h w ')
        residual = x 
        x = self.conv1x1_1(x)
        x = self.gelu(x)
        x = self.dw_conv1(x)
        x = self.pointwise_conv1(x)

        x = self.dw_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.conv1x1_2(x)
        batch_size, channels, height, width = x.size()
        x = x.flatten(2).transpose(1, 2) 
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = x + residual  
        x = rearrange(x,'b c h w -> b w h c ')
        return x   
    
if __name__ == "__main__":
   
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    net = VUGA().to(device=device)

    test_transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = MyDataset('/mnt/10T/WKC/Databases/JUFE_10K/final_dis_10320','/mnt/10T/WKC/Databases/JUFE_10K/jufe_10k.csv',mode='test',transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=False,
    )
    text = torch.randn(1,512).to(device=device)
    for imgs, mos in test_loader:
        imgs = imgs.to(device=device)
        out = net(imgs)
        print(out)
        print(mos)
        break
    
    
    