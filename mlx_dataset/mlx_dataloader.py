
from .aug import ImgAugTransform

def dataloader(train_dataset, vaild_dataset, batch_size):
    aug_fn = ImgAugTransform()
    train_dataloader = (
        train_dataset
        .shuffle()
        .to_stream() # <-- making a stream from the shuffled buffer
        .load_numpy("x_file", output_key="image")
        .load_numpy("y_file", output_key="label")
        .key_transform("image", lambda x: x * 255)
        .key_transform("image", aug_fn)

        .batch(batch_size)
        .prefetch(8, 8)
    )

    vaild_dataloader = (
        vaild_dataset
        # .shuffle()
        .to_stream() # <-- making a stream from the shuffled buffer
        .load_numpy("x_file", output_key="image")
        .load_numpy("y_file", output_key="label")

        .batch(batch_size)
        .prefetch(8, 8)
    )

    return train_dataloader, vaild_dataloader