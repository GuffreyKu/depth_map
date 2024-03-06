import matplotlib.pyplot as plt

def drow(losses:dict, name:str):

    train_losses = losses["train"]
    valid_losses = losses["valid"]

    plt.figure(figsize=(15, 7))

    # 繪製 Training loss 和 Validation loss
    plt.subplot(121)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.legend(loc='upper left')
    plt.title('Loss')

    plt.savefig("eval_fig/%s.png"%name)