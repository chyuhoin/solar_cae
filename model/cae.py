"""
卷积自编码器模型结构，通道压缩但空间分辨率不变。
"""
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    基本卷积块，包含卷积、BN和激活。
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, act=nn.GELU, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act())
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    残差块，支持空洞卷积。
    """
    def __init__(self, ch, act=nn.GELU, bn=True, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation, bias=not bn)
        self.bn1 = nn.BatchNorm2d(ch) if bn else nn.Identity()
        self.act = act()
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation, bias=not bn)
        self.bn2 = nn.BatchNorm2d(ch) if bn else nn.Identity()
    def forward(self, x):
        y = self.conv1(x); y = self.bn1(y); y = self.act(y)
        y = self.conv2(y); y = self.bn2(y)
        return self.act(x + y)

class ChannelCAE(nn.Module):
    """
    通道压缩型卷积自编码器。
    编码器：卷积+残差堆叠，最后1x1卷积降维到latent通道。
    解码器：对称结构，最后1x1卷积恢复原通道+Sigmoid。
    空间分辨率保持不变。
    """
    def __init__(self, in_ch=118, latent_ch=16, mid_ch=128, depth=4, dilations=(1,2,4,8)):
        super().__init__()
        self.in_ch = in_ch
        self.latent_ch = latent_ch
        self.mid_ch = mid_ch
        self.depth = depth
        if isinstance(dilations, int):
            dilations = (dilations,) * depth
        layers = [ConvBlock(in_ch, mid_ch, k=3, s=1, p=1)]
        for i in range(depth):
            d = dilations[i % len(dilations)]
            layers.append(ResidualBlock(mid_ch, dilation=d))
        self.enc = nn.Sequential(*layers)
        self.to_latent = nn.Conv2d(mid_ch, latent_ch, kernel_size=1)
        dlayers = []
        dlayers.append(nn.Conv2d(latent_ch, mid_ch, kernel_size=1))
        dlayers.append(nn.GELU())
        for i in range(depth):
            d = dilations[(depth-1-i) % len(dilations)]
            dlayers.append(ResidualBlock(mid_ch, dilation=d))
        self.dec = nn.Sequential(*dlayers)
        self.out_conv = nn.Conv2d(mid_ch, in_ch, kernel_size=1)
        self.out_act = nn.Sigmoid()
    def encode(self, x):
        x = self.enc(x)
        z = self.to_latent(x)
        return z
    def decode(self, z):
        y = self.dec(z)
        y = self.out_conv(y)
        y = self.out_act(y)
        return y
    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y, z
