import mlx.nn as nn

def loss_fn(model, x, y):
    logits = model(x)
    loss1 = nn.losses.huber_loss(logits, y, reduction='mean')
    loss2 = nn.losses.cosine_similarity_loss(logits, y, reduction='mean')
    return loss1+((1-loss2)*0.5)