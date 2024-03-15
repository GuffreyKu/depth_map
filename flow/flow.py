import mlx.core as mx
from tqdm import tqdm
import numpy as np

def train(dataloader,
          step_fn, 
          input_key:str, 
          label_key:str,
          now_ep:int
          ):

    b_train_loss = []

    with tqdm(dataloader) as loader:
        for batch in loader:
            loader.set_description(f"train {now_ep}")
            x = mx.array(batch[input_key])
            y = mx.array(batch[label_key])
            loss, lr = step_fn(x, y)
            
            b_train_loss.append(loss.item())

            loader.set_postfix(loss=np.mean(b_train_loss), lr=lr.item())

        return np.mean(b_train_loss).astype("float16")


def evaluate(dataloader,
             model,
             loss_fn,
             input_key:str,
             label_key:str
             ):
    
    b_valid_loss = []
    with tqdm(dataloader) as loader:
        for batch in loader:
            loader.set_description("valid")

            x = mx.array(batch[input_key])
            y = mx.array(batch[label_key])

            loss = loss_fn(model, x, y)
            
            b_valid_loss.append(loss.item())
            loader.set_postfix(loss=np.mean(b_valid_loss))

        return np.mean(b_valid_loss).astype("float16")