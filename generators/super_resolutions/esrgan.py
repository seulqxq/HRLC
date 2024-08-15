import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

# 空间特征变换  处理深度图信息 深度调制 公式8
'''
scale: γ
shift: β
x: 输入特征 F
cond: 条件信息 C (深度图)
'''
class SFTLayer(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(gc, gc, 1)       # 计算空间特征变换中的缩放因子
        self.SFT_scale_conv1 = nn.Conv2d(gc, nf, 1)
        self.SFT_shift_conv0 = nn.Conv2d(gc, gc, 1)       # 计算空间特征变换中的偏移量
        self.SFT_shift_conv1 = nn.Conv2d(gc, nf, 1)

    def forward(self, x, cond):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.2, inplace=True))   # γ
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.2, inplace=True))   # β
        return x * (scale + 1) + shift


class ResidualDenseBlock_SFT(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        nf (int): Channel number of intermediate features.
        gc (int): Channels for each growth.
    """

    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock_SFT, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.sft0 = SFTLayer(nf, gc)
        self.sft1 = SFTLayer(gc, gc)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
    
    # x --> [feature map, condition map]
    def forward(self, x):
        xc0 = self.sft0(x[0], x[1])
        x1 = self.lrelu(self.conv1(xc0))
        x2 = self.lrelu(self.conv2(torch.cat((xc0, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((xc0, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((xc0, x1, x2, x3), 1)))
        xc1 = self.sft1(x4, x[1])
        x5 = self.conv5(torch.cat((xc0, x1, x2, x3, xc1), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return (x5 * 0.2 + x[0], x[1])


class RRDB_SFT(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        nf (int): Channel number of intermediate features.
        gc (int): Channels for each growth.
    """

    def __init__(self, nf, gc=32):
        super(RRDB_SFT, self).__init__()
        self.rdb1 = ResidualDenseBlock_SFT(nf, gc)
        self.rdb2 = ResidualDenseBlock_SFT(nf, gc)
        self.rdb3 = ResidualDenseBlock_SFT(nf, gc)
        self.sft0 = SFTLayer(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.sft0(out[0], x[1])
        # Emperically, we use 0.2 to scale the residual for better performance
        return (out * 0.2 + x[0], x[1])

class RRDBNet(nn.Module):
    """
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        nf (int): Channel number of intermediate features.
            Default: 64
        nb (int): Block number in the trunk network. Defaults: 23
        gc (int): Channels for each growth. Default: 32.
    """
    # in_nc = 3
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale_factor=4, num_cond=1, up_channels=None, to_rgb_ks=3, use_pixel_shuffle=True, interpolate_mode='bilinear', global_residual=False):
        super(RRDBNet, self).__init__()
        self.scale_factor = scale_factor
        self.use_pixel_shuffle = use_pixel_shuffle
        self.global_residual = global_residual
        self.interpolate_mode = interpolate_mode
        # if scale == 2:
        #     num_in_ch = num_in_ch * 4
        # elif scale == 1:
        #     num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = make_layer(RRDB_SFT, nb, nf=nf, gc=gc) # DM-RRDB ×5
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        # if in_nc > 3:
        #     self.conv_fea = nn.Conv2d(in_nc, nf, 3, 1, 1)
        #     self.conv_prefea = nn.Conv2d(2*nf, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sftbody = SFTLayer(nf, gc)
        self.CondNet = nn.Sequential(
            nn.Conv2d(num_cond, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 1))
        
        #### upsampling
        self.num_upconvs = int(np.log2(scale_factor))
        if up_channels is None:
            up_channels = [nf] * 2 + [nf // 2] * (self.num_upconvs - 2)
        self.upconvs = nn.ModuleList([])
        if self.use_pixel_shuffle:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(nn.Conv2d(nf, up_channels[0] * 4, 3, 1, 1, bias=True))
                else:
                    self.upconvs.append(nn.Conv2d(up_channels[i - 1], up_channels[i] * 4, 3, 1, 1, bias=True))
        else:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(nn.Conv2d(nf, up_channels[0], 3, 1, 1, bias=True))
                else:
                    self.upconvs.append(nn.Conv2d(up_channels[i - 1], up_channels[i], 3, 1, 1, bias=True))
                   
        self.HRconv = nn.Conv2d(up_channels[-1], up_channels[-1], 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(up_channels[-1], out_nc, to_rgb_ks, 1, (to_rgb_ks-1)//2, bias=True)
        if self.global_residual:
            with torch.no_grad():
                self.conv_last.weight *= 0
                self.conv_last.bias *= 0
                
    # x: feature map cond: depth map
    def forward(self, x, cond):

        feat = self.conv_first(x)
        # else:
        #     feat_rgb = self.conv_first(x)
        #     # feat += torch.sigmoid(self.conv_fea(fea))
        #     feat = torch.cat((feat_rgb, fea), dim=1)
        #     feat = self.conv_prefea(feat)
        cond = self.CondNet(cond)   # 处理条件信息（深度图）
        body_feat = self.body((feat, cond)) # DM-RRDB ×5
        body_feat = self.sftbody(body_feat[0], body_feat[1])    # 空间特征变换 F = γ * F + β
        body_feat = self.conv_body(body_feat)
        body_feat += feat       # 残差连接
        # upsample
        # if self.scale > 1:
        #     body_feat = self.lrelu(self.conv_up1(F.interpolate(body_feat, scale_factor=2, mode='nearest')))
        #     if self.scale == 4:
        #         body_feat = self.lrelu(self.conv_up2(F.interpolate(body_feat, scale_factor=2, mode='nearest')))
        # out = self.conv_last(self.lrelu(self.conv_hr(body_feat)))
        
        if self.use_pixel_shuffle:
            for i in range(self.num_upconvs):
                body_feat = self.lrelu(F.pixel_shuffle(self.upconvs[i](body_feat), 2))
        else:
            for i in range(self.num_upconvs):
                body_feat = F.interpolate(self.lrelu(self.upconvs[i](body_feat)), scale_factor=2, mode=self.interpolate_mode)

        out = self.conv_last(self.lrelu(self.HRconv(body_feat)))
        # print("out.shape: ", out.shape)
        # print("x.shape: ", x.shape)
        assert out.shape[-1] == x.shape[-1] * self.scale_factor

        if self.global_residual:
            out = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic') + 0.2 * out
        return out
