"""
Implementation of ResNets for CIFAR-10 as per the original paper [https://arxiv.org/abs/1512.03385].
Configurations include ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110, ResNet-1202.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

class ShortcutA(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x):
        return mx.pad(
            x[:, ::2, ::2, :],
            pad_width=[(0, 0), (0, 0), (0, 0), (self.dims // 4, self.dims // 4)],
        )


class Block(nn.Module):
    """
    Implements a ResNet block with two convolutional layers and a skip connection.
    As per the paper, CIFAR-10 uses Shortcut type-A skip connections. (See paper for details)
    """
    def __init__(self, in_dims, dims, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(dims)

        self.conv2 = nn.Conv2d(
            dims, dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(dims)

        init_fn = nn.init.he_normal()
        self.conv1.weight = init_fn( self.conv1.weight)
        self.conv2.weight = init_fn( self.conv2.weight)

        if stride != 1:
            self.shortcut = ShortcutA(dims)
        else:
            self.shortcut = None

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is None:
            out += x
        else:
            out += self.shortcut(x)
        out = nn.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(16)

        self.layer1 = self._make_layer(block, 16, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, 64, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 64, 128, num_blocks[2], stride=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.conv_cat1 = nn.Conv2d(
            192, 64, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.conv_cat2 = nn.Conv2d(
            96, 64, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.conv_cat3 = nn.Conv2d(
            96, 64, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.output = nn.Conv2d(
            64, 1, kernel_size=1, stride=1, padding=0, bias=True
        )

        init_fn = nn.init.he_normal()
        self.conv1.weight = init_fn( self.conv1.weight)
        self.conv_cat1.weight = init_fn( self.conv_cat1.weight)
        self.conv_cat2.weight = init_fn( self.conv_cat2.weight)
        self.conv_cat3.weight = init_fn( self.conv_cat3.weight)
        self.output.weight = init_fn( self.output.weight)


    def _make_layer(self, block, in_dims, dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims, dims, stride))
            in_dims = dims
        return nn.Sequential(*layers)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def __call__(self, x):
        # encoder
        x = nn.relu(self.bn1(self.conv1(x)))
        c = x
        x = self.layer1(x)
        c0 = x
        x = self.layer2(x)
        c1 = x
        x = self.layer3(x)
        c2 = x
        x = self.layer4(x)

        # decoder
        x = self.upsample(x)
        x = mx.concatenate([x, c2], axis=-1)
        x = self.conv_cat1(x)
        x = self.upsample(x)    
        x = mx.concatenate([x, c1], axis=-1)
        x = self.conv_cat2(x)
        x = self.upsample(x)
        x = mx.concatenate([x, c0, c], axis=-1)
        x = self.conv_cat3(x)
        x = self.upsample(x)

        x = self.output(x)

        return x


def resnet20(**kwargs):
    return ResNet(Block, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(Block, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(Block, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(Block, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(Block, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return ResNet(Block, [200, 200, 200], **kwargs)

    
if __name__ == "__main__":
    import numpy as np

    input = mx.array(np.random.rand(16, 128, 256, 3))
   
    model = resnet20()
    pred = model(input)

    print(pred.shape)