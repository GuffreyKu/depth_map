import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.data as dx
from mlx.utils import tree_flatten
from functools import partial
import numpy as np

from mlx_dataset.mlx_dataset import collect_data
from mlx_dataset.mlx_dataloader import dataloader
from model.model import resnet20
from loss.loss import loss_fn
from utils.tool import  folder_check, EarlyStopping
from flow.flow import train, evaluate
from utils.loss_vis import drow


train_data_path = "data/train/"
valid_data_path = "data/val/"

epoch = 100
batch_size = 64

lr_schedule = optim.cosine_decay(1e-3, batch_size*epoch)
optimizer = optim.Adam(learning_rate=lr_schedule)

model = resnet20()

early_stopping = EarlyStopping( patience=5,
                                verbose=False)

state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss), grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

if __name__ == "__main__":

    folder_check()

    train_dataset = dx.buffer_from_vector(collect_data(train_data_path))
    vaild_dataset = dx.buffer_from_vector(collect_data(valid_data_path))

    print( "Train data size : ", len(train_dataset))
    print( "Valid data size : ", len(train_dataset))

    train_dataloader, vaild_dataloader = dataloader(train_dataset=train_dataset, 
                                                    vaild_dataset=vaild_dataset,
                                                    batch_size=batch_size)

    best_loss = np.Inf

    ep_losses = {"train":[], "valid":[]}

    for e in range(epoch):
        b_train_loss = train(dataloader=train_dataloader, 
                             step_fn=step,
                             input_key="image",
                             label_key="label")
        
        mx.eval(state)
        b_valid_loss = evaluate(dataloader=vaild_dataloader, 
                                model=model,
                                loss_fn=loss_fn,
                                input_key="image",
                                label_key="label")

        ep_train_loss = np.mean(b_train_loss)
        ep_valid_loss = np.mean(b_valid_loss)

        ep_losses["train"].append(ep_train_loss)
        ep_losses["valid"].append(ep_valid_loss)

        if ep_valid_loss <= best_loss:
            best_loss = ep_valid_loss
            flat_params = tree_flatten(model.parameters())
            mx.savez("savemodel/model.npz", **dict(flat_params))

        early_stopping(ep_valid_loss)

        if early_stopping.early_stop:
            print("!! Early Stopping")
            break

        print(
            f"Epoch {e}: Train loss : {ep_train_loss}, "
            f"Valid loss :  {ep_valid_loss},"
        )

        train_dataloader.reset()
        vaild_dataloader.reset()

    drow(losses=ep_losses,
         name="loss")