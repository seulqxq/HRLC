import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def make_model(scale_factor, parent=False):
    return AAFFBlk(scale_factor)

wn = lambda x: torch.nn.utils.weight_norm(x)

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class CALayer(nn.Module):
    def __init__(self, channel, reduction=1, use_hsigmoid=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_hsigmoid: # use hsigmoid instead of sigmoid
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                hsigmoid())
        else:
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())
    
    def forward(self, x):
        y = self.avg_pool(x)
        #print(y.size())
        y = self.conv_du(y)
        return x * y

class Block_a(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn=None, act=nn.ReLU(True)):
        super(Block_a, self).__init__()
        body = []
        conv1 = nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2, bias=True)
        conv2 = nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        # if wn is not None:
        #     conv1 = wn(conv1)
        #     conv2 = wn(conv2)
        body.append(conv1)
        body.append(act)
        body.append(conv2)
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) 
        return res

class create_model(nn.Module):
    def __init__(self, scale_factor):
        super(create_model, self).__init__()
        self.scale_factor = scale_factor
        self.model = make_model(scale_factor=scale_factor)
    
    def forward(self, x): 
        return self.model(x)

'''
input: lr_pixel [B, 3, HL, HL]
output: hr_pixel [B, 3, HL*scale_factor, HL*scale_factor]
'''
class AAFFBlk(nn.Module):
    def __init__(self, scale_factor=4, in_nc=3, res_s=1, n_resblocks=8, n_feats=16, block_feats=128, use_hsigmoid=False, use_ca=True):
        super(AAFFBlk, self).__init__()
        self.scale_factor = scale_factor
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = []
        head.append(wn(nn.Conv2d(in_nc, n_feats, 3, padding=3//2, bias=True)))

        # define body module
        body = []
        self.x_scale_list = nn.ModuleList()
        self.res_scale_list = nn.ModuleList()
        self.auxilary_scale_list = nn.ModuleList()
        for _ in range(n_resblocks):
            body.append(Block_a(n_feats, 3, block_feats))
            self.x_scale_list.append(Scale(res_s))
            self.res_scale_list.append(Scale(res_s))
            self.auxilary_scale_list.append(Scale(res_s))
                    
        # define tail module
        out_feats = scale_factor*scale_factor*in_nc
        tail = []
        tail.append(
            nn.Conv2d(n_feats, out_feats, 3, padding=3//2, bias=True)
        )
        tail.append(nn.PixelShuffle(scale_factor))

        # define skip module
        skip = []
        skip.append(
            nn.Conv2d(in_nc, out_feats, 3, padding=3//2, bias=True)
        )
        skip.append(nn.PixelShuffle(scale_factor))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # auxilary features
        self.fusion_conv_list = nn.ModuleList()
        for j in range(n_resblocks):
            if use_ca:
                tmp = nn.Sequential(*[nn.Conv2d(n_feats*(j+1), n_feats, 1, padding=0, bias=False), CALayer(n_feats, 1, use_hsigmoid=use_hsigmoid)])
            else:
                tmp = nn.Sequential(*[nn.Conv2d(n_feats*(j+1), n_feats, 1, padding=0, bias=False)])
            self.fusion_conv_list.append(tmp)

        self.test_flops = False
        
    def forward(self, x):
        # if self.test_flops:
        #     x = (x - self.rgb_mean*255)/127.5 
        # else:
        #     x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        s0 = s1 = self.head(x)

        state_list = []
        state_list.append(s0)

        for i, blo in enumerate(self.body):
            s0, s1 = s1, blo(s1)
            s1 = self.x_scale_list[i](s0) + self.res_scale_list[i](s1)
            s1 = s1 + self.auxilary_scale_list[i](self.fusion_conv_list[i](torch.cat(state_list, dim=1)))
            state_list.append(s1)

        out = self.tail(s1)
        out = out + s
        
        # if self.test_flops:
        #     # out = out*127.5 + self.rgb_mean*255
        #     out = out + self.rgb_mean
        # else:
        #     # out = out*127.5 + self.rgb_mean.cuda()*255
        #     out = out + self.rgb_mean.cuda()
        return out