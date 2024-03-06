import mlx.core as mx
import mlx.data as dx

def train(dataloader,
          step_fn, 
          input_key:str, 
          label_key:str
          ):

    b_train_loss = []

    for batch in dataloader:
        x = mx.array(batch[input_key])
        y = mx.array(batch[label_key])
        loss = step_fn(x, y)

        b_train_loss.append(loss.item())

    return b_train_loss


def evaluate(dataloader,
             model,
             loss_fn,
             input_key:str,
             label_key:str
             ):
    
    b_valid_loss = []
    for batch in dataloader:
            x = mx.array(batch[input_key])
            y = mx.array(batch[label_key])
            loss = loss_fn(model, x, y)

            b_valid_loss.append(loss.item())

    return b_valid_loss