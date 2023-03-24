# -*- coding: utf-8 -*-


# ***************************************************
# * File        : bert_use.py
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

from bert_serving.client import BertClient


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


bc = BertClient(
    ip = "localhost", 
    check_version = False, 
    check_length = False
)
vec = bc.encode(["学习"])
print(vec)








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
