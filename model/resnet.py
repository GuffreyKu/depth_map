import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import sys
import os
from pathlib import Path
WORKING_DIRECTORY = Path(__file__).parent.resolve()
sys.path.append(os.path.join(str(WORKING_DIRECTORY / '.')))
from module import ResidualIdentity, ResidualIConv
    
class ResNet(nn.Module):
    def __init__(self, num_blocks, channel_blocks, attention_blocks, act=nn.Mish()):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channel_blocks[0], kernel_size=7, stride=2, padding=3,bias=False),
                                  nn.BatchNorm(channel_blocks[0]),
                                  self.act)
        #Residual Block 1
        #(64, 56, 56) -> (64, 28, 28)
        self.first_stage = self.block(num_blocks[0], 
                                      in_channel = channel_blocks[0], 
                                      out_channel = channel_blocks[0],
                                      attention = attention_blocks[0])
        #Residual Block 2
        #(64, 28, 28) -> (128, 14, 14)
        self.second_stage = self.block(num_blocks[1], 
                                      in_channel = channel_blocks[0]*4, 
                                      out_channel = channel_blocks[1],
                                      attention = attention_blocks[1])
        #Residual Block 3
        #(128, 14, 14) -> (256, 7, 7)
        self.thrid_stage = self.block(num_blocks[2], 
                                      in_channel = channel_blocks[1]*4, 
                                      out_channel = channel_blocks[2],
                                      attention = attention_blocks[2])
        #Residual Block 4
        #(256, 7, 7) -> (512, 7, 7)
        self.forth_stage = self.block(num_blocks[3], 
                                      in_channel = channel_blocks[2]*4, 
                                      out_channel = channel_blocks[3],
                                      attention = attention_blocks[3],
                                      down_sample=False)
        
        self.output = nn.Conv2d(
            channel_blocks[3]*4, 1, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams
        
    def block(self, num, in_channel, out_channel, attention, down_sample=True):
        layers = list()
        
        # input channel = 64, output channel = 64 * 4(僅需要輸入input_channel數量即可)
        layers.append(ResidualIConv(in_channels = in_channel, out_channels = out_channel))
        for _ in range(num):
            layers.append(ResidualIdentity(in_channels = out_channel*4, out_channels = out_channel, attention=attention))
        if down_sample:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    
    def __call__(self, x, iscam=False):
        x = self.conv(x)
        x = self.first_stage(x)
        x = self.second_stage(x)
        x = self.thrid_stage(x)
        x = self.forth_stage(x)

        x = self.upsample(x)
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.output(x)
        return x
       

def resnet20(**kwargs):
    return ResNet([3,3,3,3], [16, 32, 64, 128], **kwargs)

def resnet56(**kwargs):
    return ResNet([3, 3, 9, 3], [64, 128, 256, 512], **kwargs)

def resnet110(**kwargs):
    return ResNet([3, 6, 18, 6], [64, 128, 256, 512], **kwargs)


if __name__ == "__main__":
    import numpy as np

    input = mx.array(np.random.rand(4, 224, 224, 3))

    model = resnet20(attention_blocks = [False, False, False, False])
    pred = model(input)

    print(pred.shape)