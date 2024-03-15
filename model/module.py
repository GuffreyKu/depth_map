import math
import mlx.core as mx
import mlx.nn as nn
from einops import rearrange, repeat
import numpy as np

class Global_Context_Block(nn.Module):
    def __init__(self, out_channels, reduction_ratio = 16):
        super(Global_Context_Block, self).__init__()

        self.context_conv = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1)
        
        self.se = nn.Sequential(nn.Conv2d(in_channels=out_channels, 
                                          out_channels=out_channels // reduction_ratio, 
                                          kernel_size=1, 
                                          stride=1,),
                                nn.BatchNorm(out_channels // reduction_ratio),)
        self.se_out = nn.Conv2d(in_channels=out_channels // reduction_ratio, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                stride=1,)
        
    def __call__(self, x):
        input_tensor = x
        # context
        b_input, h_input, w_input, c_input = input_tensor.shape
        input_tensor = input_tensor.reshape(b_input, h_input*w_input, c_input)
        x = self.context_conv(x)
        b_x, h_x, w_x, c_x = x.shape
        x = x.reshape(b_x, c_x, h_x*w_x)
        x = mx.softmax(x, axis=1)
        # (b,1,hw)@(b,hw,c) -> (b,1,c)
        out = x @ input_tensor

        b_out, hw_out, c_out = out.shape
        out = out.reshape(b_out, hw_out, hw_out, c_out)

        # SE
        out = self.se_out(nn.silu(self.se(out)))
        out = mx.sigmoid(out)
        # skip connection
        input_tensor = input_tensor.reshape(b_input, h_input, w_input, c_input)
        output = input_tensor + out
        
        return output
    
class ResidualIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.Mish(), attention=False):
        super().__init__()  
        self.act = act
        self.attention = attention
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=1, 
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=in_channels, 
                               kernel_size=1, 
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm(in_channels)
        if self.attention:
            self.att = Global_Context_Block(out_channels=in_channels, reduction_ratio = 8)

     
    def __call__(self, x):
        short_cut = x
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.attention:
            x = self.att(x)
        output = x + short_cut
        
        return self.act(output)

class ResidualIConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.Mish()):
        super().__init__()
        self.act = act
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1,
                                             stride=1,
                                             bias=False),
                                             nn.BatchNorm(out_channels),
                                   self.act,
                                   nn.Conv2d(in_channels=out_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm(out_channels),
                                   self.act,
                                   nn.Conv2d(in_channels=out_channels,
                                             out_channels=out_channels*4,
                                             kernel_size=1,
                                             stride=1,
                                             bias=False),
                                   nn.BatchNorm(out_channels*4),
                                   self.act)
        
        self.short_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels*4, 
                                    kernel_size=1, 
                                    stride=1,
                                    bias=False),
                                    nn.BatchNorm(out_channels*4))
     
    def __call__(self, x):
        out = self.conv1(x)
        x = self.short_conv(x)
        
        out += x
        out = self.act(out)
        
        return out

class ShortcutA(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x):
        return mx.pad(
            x[:, ::2, ::2, :],
            pad_width=[(0, 0), (0, 0), (0, 0), (self.dims // 4, self.dims // 4)],
        )
