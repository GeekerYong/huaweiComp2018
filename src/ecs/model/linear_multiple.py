# -*- coding:utf-8 -*-
#  @Time : 2018-03-27 16:26
#  @Author : Liu JinYong
#  @File : model_lab.py
#  @Description:
#
#   --Input:
#   --Output:

def zeros(shape):
    row = shape[0]
    col = shape[1]
    mat = []
    for i in range(row):
        row_init = []
        for j in range(col):
            row_init.append(0)
        mat.append(mat)
    return 0

def featureNormalize(X):
