import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.data as dx
from mlx.utils import tree_flatten
from functools import partial
import numpy as np

from mlx_dataset.mlx_dataset import collect_cityspace
from mlx_dataset.mlx_dataloader import cityspaceloader
from model.model import resnet20
from loss.loss import loss_fn
from utils.tool import  folder_check, EarlyStopping
from flow.flow import train, evaluate
from utils.loss_vis import drow


train_data_path = "data/train.csv"
valid_data_path = "data/val.csv"
epochs = 100
batch_size = 8

train_dataset = dx.buffer_from_vector(collect_cityspace(train_data_path))
vaild_dataset = dx.buffer_from_vector(collect_cityspace(valid_data_path))

print(vaild_dataset)
print( "Train data size : ", len(train_dataset))
print( "Valid data size : ", len(vaild_dataset))


warmup = optim.linear_schedule(0, 1e-3, steps=int(epochs*0.1) * (int(len(train_dataset) // batch_size)))
cosine = optim.cosine_decay(1e-3, epochs * (int(len(train_dataset) // batch_size)))
lr_schedule = optim.join_schedules([warmup, cosine], [int(epochs*0.1) * (int(len(train_dataset) // batch_size))])

optimizer = optim.AdamW(learning_rate=lr_schedule)

model = resnet20()


early_stopping = EarlyStopping( patience=10,
                                verbose=False)

state = [model.state, optimizer.state, mx.random.state]
@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    (loss), grads = loss_and_grad_fn(model, x, y)
    lr = optimizer.learning_rate
    optimizer.update(model, grads)
    return loss, lr

if __name__ == "__main__":

    folder_check()

    train_dataloader, vaild_dataloader = cityspaceloader(train_dataset=train_dataset, 
                                                    vaild_dataset=vaild_dataset,
                                                    batch_size=batch_size,
                                                    image_size=(480, 320))

    best_loss = np.Inf

    ep_losses = {"train":[], "valid":[]}

    for e in range(epochs):
        train_loss = train(dataloader=train_dataloader, 
                             step_fn=step,
                             input_key="image",
                             label_key="label",
                             now_ep=e)
        
        mx.eval(state)
        valid_loss = evaluate(dataloader=vaild_dataloader, 
                                model=model,
                                loss_fn=loss_fn,
                                input_key="image",
                                label_key="label")


        ep_losses["train"].append(train_loss)
        ep_losses["valid"].append(valid_loss)

        if valid_loss <= best_loss:
            best_loss = valid_loss
            flat_params = tree_flatten(model.parameters())
            mx.savez("savemodel/model.npz", **dict(flat_params))

        early_stopping(valid_loss)

        if early_stopping.early_stop:
            print("!! Early Stopping")
            break

        train_dataloader.reset()
        vaild_dataloader.reset()

    drow(losses=ep_losses,
         name="loss")