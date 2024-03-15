import mlx.nn as nn
import mlx.core as mx

def loss_fn(model, x, y):
    output = model(x)
    loss1 = nn.losses.huber_loss(output, y, reduction='mean')
    return loss1