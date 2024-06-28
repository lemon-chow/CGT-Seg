""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .unet_parts import DoubleConv, Down, Up, OutConv
from .swin_transformer import SwinTransformerBlock, Conv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class SEUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SEUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.se1 = SEBlock(128)  # 添加SE Block
        self.down2 = (Down(128, 256))
        self.se2 = SEBlock(256)  # 添加SE Block
        self.down3 = (Down(256, 512))
        self.se3 = SEBlock(512)  # 添加SE Block
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.se4 = SEBlock(1024 // factor)  # 添加SE Block
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.se_up1 = SEBlock(512 // factor)  # 添加SE Block
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.se_up2 = SEBlock(256 // factor)  # 添加SE Block
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.se_up3 = SEBlock(128 // factor)  # 添加SE Block
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.se1(x2)  # 添加SE Block
        x3 = self.down2(x2)
        x3 = self.se2(x3)  # 添加SE Block
        x4 = self.down3(x3)
        x4 = self.se3(x4)  # 添加SE Block
        x5 = self.down4(x4)
        x5 = self.se4(x5)  # 添加SE Block
        x = self.up1(x5, x4)
        x = self.se_up1(x)  # 添加SE Block
        x = self.up2(x, x3)
        x = self.se_up2(x)  # 添加SE Block
        x = self.up3(x, x2)
        x = self.se_up3(x)  # 添加SE Block
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# SE注意力机制
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(f"Input size to SEBlock: {x.size()}")
        y = self.avg_pool(x).view(b, c)
        # print(f"Size after avg_pool: {y.size()}")
        y = self.fc(y).view(b, c, 1, 1)
        # print(f"Size after fc: {y.size()}")
        return x * y.expand_as(x)

class BasicBlock(nn.Module):          
    expansion = 1  # 通道扩充比例
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
    
class BottleNeck(nn.Module):
    expansion = 4
    '''
    expansion 是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''
    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class C2fST(C2f):
    """C2f module with Swin TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        num_heads = self.c // 32
        self.m = nn.ModuleList(SwinTransformerBlock(self.c, self.c, num_heads, n) for _ in range(n))

    
class UResnet(nn.Module):
    def __init__(self, block, layers, n_channels, n_classes, bilinear):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.trans1 = SwinTransformerBlock(nb_filter[1], nb_filter[1], num_heads = 4, num_layers = 2)        
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.trans2 = SwinTransformerBlock(nb_filter[2], nb_filter[2], num_heads = 8, num_layers = 2)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.trans3 = SwinTransformerBlock(nb_filter[3], nb_filter[3], num_heads = 16, num_layers = 2)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)
        self.trans4 = SwinTransformerBlock(nb_filter[4], nb_filter[4], num_heads = 32, num_layers = 2)

        self.conv3_1 = DoubleConv((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3] * block.expansion, nb_filter[3])
        self.conv2_2 = DoubleConv((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2] * block.expansion, nb_filter[2])
        self.conv1_3 = DoubleConv((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1] * block.expansion, nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        """
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        # for stride in strides:
        #     layers.append(block(self.in_channel, middle_channel, stride))
        #     self.in_channel = middle_channel * block.expansion
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion  # 更新输入通道数为当前层的输出通道数        
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.trans1(x1_0)
        # print("conv1_0:", x1_0.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.trans2(x2_0)
        # print("conv2_0:", x2_0.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.trans3(x3_0)
        # print("conv3_0:", x3_0.shape)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.trans4(x4_0)
        # print("conv4_0:", x4_0.shape)

        # x4_0 = self.trans1(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class UResnet34(UResnet):
    def __init__(self, n_channels, n_classes=2, bilinear=False):
        super(UResnet34, self).__init__(block=BasicBlock,layers=[3,4,6,3], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)


class UResnet50(UResnet):
    def __init__(self, n_channels, n_classes=2, bilinear=False):
        super(UResnet50, self).__init__(block=BottleNeck,layers=[3,4,6,3], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)


class UResnet101(UResnet):
    def __init__(self, n_channels, n_classes=2, bilinear=False):
        super(UResnet101, self).__init__(block=BottleNeck,layers=[3,4,23,3], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)


class UResnet152(UResnet):
    def __init__(self, n_channels, n_classes=2, bilinear=False):
        super(UResnet152, self).__init__(block=BottleNeck,layers=[3,8,36,3], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)