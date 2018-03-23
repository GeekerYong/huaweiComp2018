# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:26:46 2018

@author: Khazix
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.cross_validation import train_test_split


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    # 创建一个tf.constant = C (类别数)
    C = tf.constant(C, name='C')
    # 使用tf.one_hot
    one_hot_matrix = tf.one_hot(labels, C, axis=0)  # axis控制最后确定编码的按行着排列显示，还是按列排列显示
    # Create the session
    sess = tf.Session()
    # Run the session
    one_hot = sess.run(one_hot_matrix)
    # Close the session
    sess.close()

    return one_hot


def load_dataset():
    data = []
    label = []
    # 胡蜂
    for i in range(211):
        dir = './data/hu/%s.png' % i
        img = Image.open(dir)
        img_array = np.asarray(img)
        img_reshape = img_array[:, :, 2].reshape(28 * 28, 1)
        img_reshape = [int(x) for x in img_reshape]
        data.append(img_reshape)
        label.append(0)
    # 黄蜂
    for i in range(183):
        dir = './data/huang/%s.png' % i
        img = Image.open(dir)
        img_array = np.asarray(img)
        img_reshape = img_array[:, :, 2].reshape(28 * 28, 1)
        img_reshape = [int(x) for x in img_reshape]
        data.append(img_reshape)
        label.append(1)
    # 蜜蜂
    for i in range(558):
        dir = './data/mi/%s.png' % i
        img = Image.open(dir)
        img_array = np.asarray(img)
        img_reshape = img_array[:, :, 2].reshape(28 * 28, 1)
        img_reshape = [int(x) for x in img_reshape]
        data.append(img_reshape)
        label.append(2)
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=0)
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y = train_y.reshape(761, 1)
    test_y = test_y.reshape(191, 1)
    return train_X, train_y, test_X, test_y


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
X_train = X_train
X_test = X_test
## Convert training and test labels to one hot matrices
Y_train = one_hot_matrix(Y_train_orig, 3)[:, :, 0].T
Y_test = one_hot_matrix(Y_test_orig, 3)[:, :, 0].T
#
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
#

# input_images = X_train
# input_labels = Y_train
## =============================================================================
##   模型建立
## =============================================================================
## 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层的variables和ops
W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 28], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[28]))

L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
L1_relu = tf.nn.relu(L1_conv + b_conv1)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义第二个卷积层的variables和ops
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 28, 56], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[56]))

L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
L2_relu = tf.nn.relu(L2_conv + b_conv2)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 56, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 56])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output层
W_fc2 = tf.Variable(tf.truncated_normal([1024, 3], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义损失函数和优化器
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

input_count = 761

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))
    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
    batch_size = 60
    iterations = 100
    batches_count = int(input_count / batch_size)
    remainder = input_count % batch_size
    print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))
    # 执行训练迭代
    for it in range(iterations):
        for n in range(batches_count):
            train_step.run(feed_dict={x: X_train[n * batch_size:(n + 1) * batch_size],
                                      y_: Y_train[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
        if remainder > 0:
            start_index = batches_count * batch_size;
            train_step.run(feed_dict={x: X_train[start_index:input_count - 1], y_: Y_train[start_index:input_count - 1],
                                      keep_prob: 0.5})
        iterate_accuracy = 0
        if it % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
            print ('iteration %d: 训练误差: %s  测试误差： %s' % (it, train_accuracy, iterate_accuracy))
    print ('完成训练!')
