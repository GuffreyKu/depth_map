
from .aug import ImgAugTransform
import numpy as np
import mlx.core as mx

def kaggleloader(train_dataset, vaild_dataset, batch_size, image_size=(256, 128)):
    aug_fn = ImgAugTransform()
    train_dataloader = (
        train_dataset
        .shuffle()
        .to_stream() # <-- making a stream from the shuffled buffer
        .load_numpy("x_file", output_key="image")
        .load_numpy("y_file", output_key="label")
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

def depth_norm(x):
    w, h, c = x.shape
    dis = np.zeros((w, h, c))
    dis[ x > 0 ] = (x[x > 0] - 1) / 256
    return mx.array(dis)

def crop_this(image):
    image_shape = image.shape
    
    length = 224
    width = 224
    
    start_row = 0
    start_column = 20
    
    end_row = length + start_row
    end_row = end_row if end_row <= image_shape[0] else image_shape[0]
    
    end_column = width + start_column
    end_column = end_column if end_column <= image_shape[1] else image_shape[1]
    
    image = image[start_row:end_row, start_column:end_column]
    return np.ascontiguousarray(image)

def cityspaceloader(train_dataset, vaild_dataset, batch_size, image_size:tuple):
    aug_fn = ImgAugTransform()
    train_dataloader = (
        train_dataset
        .shuffle()
        .to_stream() # <-- making a stream from the shuffled buffer
        .load_image("x_file", output_key="image")
        .image_resize("image", w=image_size[0], h=image_size[1])
        .key_transform("image", aug_fn)

        .load_image("y_file", output_key="label")
        .image_resize("label", w=image_size[0], h=image_size[1])
        .key_transform("label", depth_norm )

        .key_transform("label", crop_this )
        .key_transform("image", crop_this )

        .batch(batch_size)
        .prefetch(8, 8)
    )

    vaild_dataloader = (
        vaild_dataset
        # .shuffle()
        .to_stream() # <-- making a stream from the shuffled buffer
        .load_image("x_file", output_key="image")
        .image_resize("image", w=image_size[0], h=image_size[1])
        .key_transform("image", lambda x: x / 255)

        .load_image("y_file", output_key="label")
        .image_resize("label", w=image_size[0], h=image_size[1])
        .key_transform("label", depth_norm )

        .key_transform("label", crop_this )
        .key_transform("image", crop_this )

        .batch(batch_size)
        .prefetch(8, 8)
    )

    return train_dataloader, vaild_dataloader