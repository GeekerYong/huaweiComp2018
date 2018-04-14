# -*- coding:utf-8 -*- 
#  @Time : 2018-04-12 15:26
#  @Author : Liu JinYong
#  @File : nn.py
#  @Description:
#            neural network
#   --Input:
#   --Output:
import random
import math


def tannh(Z):
    res = matrix([[]])
    res.zero(Z.dimx, Z.dimy)
    for i in range(Z.dimx):
        for j in range(Z.dimy):
            z = Z.value[i][j]
            e_p = math.exp(z)
            e_m = math.exp(-z)
            res.value[i][j] = (e_p - e_m) / (e_m + e_p+0.000001)
    return res

def siggmoid(Z):
    res = matrix([[]])
    res.zero(Z.dimx, Z.dimy)
    for i in range(Z.dimx):
        for j in range(Z.dimy):
            z = Z.value[i][j]
            e_z = math.exp(-z)
            res.value[i][j] = 1/(1+e_z)
    return res


def powwer(Z):
    res = matrix([[]])
    res.zero(Z.dimx, Z.dimy)
    for i in range(Z.dimx):
        for j in range(Z.dimy):
            z = Z.value[i][j]
            res.value[i][j] = z**2
    return res

def suum(Z):
    res = 0
    for i in range(Z.dimx):
        for j in range(Z.dimy):
            res += Z.value[i][j]
    return res


def layer_sizes(X, Y):
    n_x = X.dimx
    n_h = 4
    n_y = Y.dimx
    return (n_x, n_h, n_y)


def fill_random_num(mat):
    for i in range(mat.dimx):
        for j in range(mat.dimy):
            mat.value[i][j] = random.random() * 0.1
    return mat


def initialize_parameters(n_x, n_h, n_y,X):
    W1 = matrix([[]])
    W1.zero(n_h, n_x)
    W1 = fill_random_num(W1)
    b1 = matrix([[]])
    b1.zero(n_h, X.dimy)
    W2 = matrix([[]])
    W2.zero(n_y, n_h)
    W2 = fill_random_num(W2)
    b2 = matrix([[]])
    b2.zero(n_y, X.dimy)
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = W1 * X + b1
    A1 = tannh(Z1)
    Z2 = W2 * A1 + b2
    #A2 = siggmoid(Z2)
    A2 = Z2

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.dimy  # num of example
    cost = (1.0/(2*m))*float(suum(powwer(A2-Y)))
    return cost

def backward_propagation(parameters , cache, X, Y):
    m = X.dimy

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    temp = 1.0/m
    #dZ2
    dZ2 =A2 - Y
    #dW2
    dW2 = dZ2 * A1.transpose()
    for i in range(dW2.dimx):
        for j in range(dW2.dimy):
            dW2.value[i][j] = dW2.value[i][j]*temp
    #db2
    db2 = suum(dZ2)

    #dZ1
    dZ1 = W2.transpose()*dZ2
    aa = powwer(A1)
    for i in range(aa.dimx):
        for j in range(aa.dimy):
            aa.value[i][j] = 1-aa.value[i][j]
    dZ1 = dZ1.multiply(aa)

    #dW1
    dW1 = dZ1 * X.transpose()
    for i in range(dW1.dimx):
        for j in range(dW1.dimy):
            dW1.value[i][j] = dW1.value[i][j]*temp
    #db1
    db1 = temp*suum(dZ1)

    grads = {
        'dW1':dW1,
        'db1':db1,
        'dW2':dW2,
        'db2':db2
    }
    return grads


def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #W1
    res = matrix([[]])
    res.zero(dW1.dimx, dW1.dimy)
    for i in range(dW1.dimx):
        for j in range(dW1.dimy):
            res.value[i][j] = dW1.value[i][j]*learning_rate
    W1 = W1 - res

    #b1
    res = db1*learning_rate
    for i in range(b1.dimx):
        for j in range(b1.dimy):
            b1.value[i][j] = b1.value[i][j] - res

    #W2
    res = matrix([[]])
    res.zero(dW2.dimx, dW2.dimy)
    for i in range(dW2.dimx):
        for j in range(dW2.dimy):
            res.value[i][j] = dW2.value[i][j]*learning_rate
    W2 = W2 - res

    #b2
    res = db2*learning_rate
    for i in range(b2.dimx):
        for j in range(b2.dimy):
            b2.value[i][j] = b2.value[i][j] - res

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def nn_model(X, Y, n_h, learning_rate,num_iterations=10000, print_cost=False):

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]


    parameters = initialize_parameters(n_x, n_h, n_y , X)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 10 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    # 整理系数
    b1 = parameters['b1']
    res = matrix([[]])
    res.zero(b1.dimx,1)
    for i in range(b1.dimx):
        res.value[i][0] = b1.value[i][0]
    parameters['b1'] = res

    b2 = parameters['b2']
    res = matrix([[]])
    res.zero(b2.dimx,1)
    for i in range(b2.dimx):
        res.value[i][0] = b2.value[i][0]
    parameters['b2'] = res
    return parameters


from matrix import *

if __name__ == '__main__':
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14]
    p = 3
    mat_x = matrix([[]])
    mat_x.zero(len(samples) - p, p)
    k = 0
    for i in range(0, len(samples) - p):
        for j in range(p):
            mat_x.value[i][j] = samples[k + j]
        k = k + 1
    mat_x = mat_x.transpose()
    mat_x.show()
    mat_y = matrix([samples[p:len(samples)]])
    mat_y.show()
    X = mat_x
    Y = mat_y
    parameters = nn_model(X,Y, 4, num_iterations=10)
    pred = forward_propagation(X, parameters)
    print()
