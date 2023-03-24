# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-22
# * Version     : 0.1.032215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import random
from typing import List

import numpy as np
from data import train_data, test_data


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 词典
# ------------------------------
# 构建词典(vocabulary)
vocab = list(set([w for text in train_data.keys() for w in text.split(" ")]))
vocab_size = len(vocab)
print(f"{vocab_size} unique words found.")
# 给词典创建索引
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# ------------------------------
# input one-hot pres
# ------------------------------
def createInputs(text: str) -> List:
    """
    Returns an array of one-hot vectors representing the words
    in the input text string.

    Args:
        text (str): string

    Returns:
        List: Each one-hot vector has shape (vocab_size, 1)
    """
    inputs = []
    for w in text.split(" "):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    
    return inputs

# ------------------------------
# RNN 前向传播
# ------------------------------
def softmax(xs):
    """
    Softmax Function
    """
    return np.exp(xs) / sum(np.exp(xs))


class RNN:

    def __init__(self, input_size, output_size, hidden_size = 64) -> None:
        # weights
        self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.Why = np.random.randn(output_size, hidden_size) / 1000
        # biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        """
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.

        Args:
            inputs (_type_): is an array of one-hot vectors with shape (input_size, 1).

        Returns:
            _type_: _description_
        """
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = { 0: h }

        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Compute the output
        y = self.Why @ h + self.by

        return y, h
    
    def backprop(self, d_y, learn_rate=2e-2):
        """
        Perform a backward pass of the RNN.

        Args:
            d_y (_type_): (dL/dy) has shape (output_size, 1).
            learn_rate (float, optional): _description_. Defaults to 2e-2.
        """
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Calculate dL/dh for the last h.
        d_h = self.Why.T @ d_y

        # Backpropagate through time.
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp
            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T
            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T
            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Clip to prevent exploding gradients.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out = d)

        # Update weights and biases using gradient descent.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


rnn = RNN(vocab_size, 2)


def processData(data, backprop = True):
    """
    Returns the RNN's loss and accuracy for the given data.
        - data is a dictionary mapping text to True or False.
        - backprop determines if the backward phase should be run.

    Args:
        data (_type_):  a dictionary mapping text to True or False.
        backprop (bool, optional): backprop determines if the backward phase should be run.. Defaults to True.

    Returns:
        _type_: _description_
    """
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)
        # Forward
        out, _ = rnn.forward(inputs)
        probs = softmax(out)
        # Calculate loss / accuracy
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)
        if backprop:
            # Build dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            # Backward
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


# Training loop
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
