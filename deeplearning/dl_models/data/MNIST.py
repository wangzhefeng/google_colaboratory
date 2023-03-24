# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_load.py
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

import torchvision
from torchvision.transforms import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


train_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = True,
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ]),
    download = True,
)

test_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = False,
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
    ]),
    download = True,
)




__all__ = [
    train_dataset,
    test_dataset,
]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
