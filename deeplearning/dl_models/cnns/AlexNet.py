# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AlexNet.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032305
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
num_classes = 10
batch_size = 64
num_epochs = 20
learning_rate = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device {device}.")


# ------------------------------
# data
# ------------------------------


# ------------------------------
# model
# ------------------------------
class AlexNet(nn.Module):

    def __init__(self, num_classes) -> None:
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),  # 3@227x227 -> 96@55x55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 96@55x55 -> 96@27x27
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),  # 96@27x27 -> 256@27x27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@27x27  -> 256@13x13
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),  # 256@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 256@13x13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@13x13 -> 256@6x6
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),  # 9216 -> 4096
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # 4096 -> 4096
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),  # 4096 -> 10
        )
    
    def forward(self, x):
        """
        shape of x: 3@227x227
        """
        x = self.layer1(x)  # 3@227x227 -> 96@27x27
        x = self.layer2(x)  # 96@27x27 -> 256@13x13
        x = self.layer3(x)  # 256@13x13 -> 384@13x13
        x = self.layer3(x)  # 384@13x13 -> 384@13x13
        x = self.layer5(x)  # 384@13x13 -> 256@6x6
        x = x.reshape(x.size(0), -1)  # 256@6x6 -> 9216
        x = self.fc1(x)  # 9216 -> 4096
        x = self.fc2(x)  # 4096 -> 4096
        out = self.fc3(x)  # 4096 -> 10
        return out



# ------------------------------
# model train
# ------------------------------
# model
model = AlexNet(num_classes)

# loss
loss = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)


# total step
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        losses = loss(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, losses.item()))
 
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 


# ------------------------------
# model testing
# ------------------------------
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
