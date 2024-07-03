# Wonsang Yun, Dept. of mathmetics, Yonsei Univ.
# DEEP LEARNING (STA3140.01-00) Final project 2024/05/07~

import torch
import torch.utils
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


def get_mean_std(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
    stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    mean = [meanR, meanG, meanB]

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])

    std = [stdR, stdG, stdB]

    return mean, std

def trainLoader(BATCH_SIZE):
    # File Input, Resize, Normalization(see file_lo)
    train_path = 'train'
    size = (72, 388)
    
    resize_train_mean = [0.96593493, 0.96593493, 0.96593493]
    resize_train_std = [0.13249733, 0.13249733, 0.13249733]

    transform_train = transforms.Compose([
        transforms.Resize(size), # Image resize
        transforms.ToTensor(),
        transforms.Normalize(resize_train_mean, resize_train_std) # Normalization
    ])

    trainset = datasets.ImageFolder(root=train_path, transform=transform_train)

    val_size = 0.1
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    split = int(np.floor(val_size*len(trainset)))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            sampler=train_sampler,num_workers=0)
    
    valloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            sampler=val_sampler,num_workers=0)
    
    return trainloader, valloader, split


def testLoader(feature, BATCH_SIZE):
    test_path = 'test/' + feature
    size = (72, 388)

    if feature == 'T':
        resize_test_mean = [0.9661202, 0.9661202, 0.9661202]
        resize_test_std = [0.13271476, 0.13271476, 0.13271476]
    
    if feature == 'U':
        resize_test_mean = [0.9680696, 0.9680696, 0.9680696]
        resize_test_std = [0.12851168, 0.12851168, 0.12851168]
    
    if feature == 'S':
        resize_test_mean = [0.9660212, 0.9660212, 0.9660212]
        resize_test_std = [0.132027, 0.132027, 0.132027]

    if feature == 'I':
        resize_test_mean = [0.96662706, 0.96662706, 0.96662706]
        resize_test_std = [0.13115089, 0.13115089, 0.13115089]

    if feature == 'O':
        resize_test_mean = [0.96790284, 0.96790284, 0.96790284]
        resize_test_std = [0.12857606, 0.12857606, 0.12857606]

    if feature == 'M':
        resize_test_mean = [0.96551895, 0.96551895, 0.96551895]
        resize_test_std = [0.13397531, 0.13397531, 0.13397531]
    
    transform_test = transforms.Compose([
        transforms.Resize(size), # Image resize
        transforms.ToTensor(),
        transforms.Normalize(resize_test_mean, resize_test_std) # Normalization
    ])

    testset = datasets.ImageFolder(root = test_path, transform = transform_test)
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             num_workers=0)
    
    
    return testLoader, len(testset)