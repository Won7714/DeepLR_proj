import torch
import numpy as np

def Blur(image, epsilon, device):
    size = image.size()
    size = list(size)
    resize_train_mean = 0
    resize_train_std = 1

    blur = np.random.normal(resize_train_mean, resize_train_std, size)
    blur = torch.from_numpy(blur).float().to(device)
    blur_img = (1 - epsilon) * image + epsilon * blur

    return blur_img