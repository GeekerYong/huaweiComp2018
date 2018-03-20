# -*- coding:utf-8 -*- 
#  @Time : 2018-03-20 15:37
#  @Author : Liu JinYong
#  @File : plot.py
#  @Description:
#           This script contains the functions needed for plotting
import matplotlib.pyplot as plt
from pylab import *

def plot_flavor_data(data, title="default"):
    date= [item[0] for item in data]
    x = range(len(date))
    y = [item[1] for item in data]
    plt.plot(x, y)
    plt.xticks(x, date, rotation=45)
    plt.show()
    return None