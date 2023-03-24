# -*- coding: utf-8 -*-


# ***************************************************
# * File        : CIFAR10.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032308
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 64


# transforms
train_valid_normalize = transforms.Normalize(
    mean = [0.4914, 0.4822, 0.4465],
    std = [0.2023, 0.1994, 0.2010],
)
test_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
valid_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    train_valid_normalize,
])
train_transform_augment = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    train_valid_normalize,
])
train_transfrom = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    train_valid_normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    test_normalize,
])

# data
train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = train_transfrom,
    download = True,
)
valid_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = valid_transform,
    download = True,
)
test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = test_transform,
    download = True
)

# data split
valid_size = 0.1
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.seed(2023)
np.random.shuffle(indices)
num_valid = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    sampler = train_sampler,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size = batch_size,
    sampler = valid_sampler,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
