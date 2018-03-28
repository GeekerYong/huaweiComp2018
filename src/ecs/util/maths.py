# -*- coding:utf-8 -*- 
#  @Time : 2018-03-28 18:52
#  @Author : Liu JinYong
#  @File : mathlab.py
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
        mat.append(row_init)
    return mat


def mean(X):
    return None


def std(X):
    return None
