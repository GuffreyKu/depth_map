from model.model import resnet20

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

model = resnet20()
model.load_weights("savemodel/model.npz")
mx.eval(model.parameters())


if __name__ == "__main__":

    gt = np.load("data/val/depth/0.npy")
    img = np.load("data/val/image/0.npy")
    img = np.expand_dims(img, axis=0)
    img = mx.array(img)

    depth_map = model(img)

    img = np.array(img).squeeze()
    depth_map = np.array(depth_map).squeeze()

    ssim_score =  ssim(gt.squeeze(), np.array(depth_map), data_range=gt.max()-gt.min())
    
    print("SSIM score", ssim_score)

    plt.figure(figsize=(15, 5))

    plt.text(0.8, 0.1, "SSIM score:" +str(ssim_score))
    plt.subplot(1,3,1)

    
    plt.imshow(img)
    plt.title("Raw Image")

    plt.subplot(1,3,2)
    plt.imshow(gt)
    plt.title("Raw depth")

    plt.subplot(1,3,3)
    plt.imshow(depth_map)
    plt.title("Depth map")
    
    plt.show()
