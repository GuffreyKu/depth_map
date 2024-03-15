import imgaug.augmenters as iaa
import numpy as np
import random

class ImgAugTransform:
    def __init__(self):
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_brightness = iaa.Add((-30, 15))
        self.aug_flipub = iaa.Flipud(1.0)
        self.aug_blur = iaa.GaussianBlur(sigma=(0.1, 1.0))
        self.aug_noise = iaa.imgcorruptlike.GaussianNoise(severity=2)

    def __call__(self, img):

        aug = random.randint(0, 3)
        # img = img * 255 # for kaggle data
        img  =  img.astype("uint8")
        if aug == 0:
            aug_img = self.aug_brightness(image=img)

            aug_img = aug_img.astype(np.float32)/255

            return aug_img

        elif aug == 1:
            aug_img = self.aug_blur(image=img)
            aug_img = aug_img.astype(np.float32)/255

            return aug_img

        elif aug == 2:
            aug_img = img.astype(np.float32)/255
            return aug_img

        elif aug == 3:
            aug_img = self.aug_noise(image=img)
            aug_img = img.astype(np.float32)/255
            return aug_img
