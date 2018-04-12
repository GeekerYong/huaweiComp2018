# coding=utf-8
import preprocessing
import os
import datetime
import math
import subprocess
from matrix import *


def ar(samples, p):
    mat_x = matrix([[]])
    mat_x.zero(len(samples) - p, p)
    k = 0
    for i in range(0, len(samples) - p):
        for j in range(p):
            mat_x.value[i][j] = samples[k + j]
        k = k + 1
    mat_y = matrix([samples[p:len(samples)]])
    mat_y = mat_y.transpose()
    b1 = mat_x.transpose() * mat_x
    b2 = b1.inverse()
    b3 = b2 * mat_x.transpose()
    b4 = b3 * mat_y
    b4.show()
    cof = [x[0] for x in b4.value]
    return cof


def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        return result
    if input_lines is None:
        return result

    mission = preprocessing.preprocess_input(input_lines)
    flavor_dict = preprocessing.preprocess_ecs_info(ecs_lines, mission)
    data_dict_merge = preprocessing.merge(flavor_dict, mission)
    data_dict_filled = preprocessing.fill_data(data_dict_merge, mission)

    # 生成序列文件送入AR模型进行预测
    flavname = data_dict_filled.keys()
    AR_cof_dict = dict()
    deg = 5
    for name in flavname:
        data = data_dict_filled[name]
        series = [x[1] for x in data]
        print("%s 预测系数生成成功" % name)
        if deg <= len(series) / 2:
            cof = ar(series, deg)
        else:
            cof = [0] * deg
        AR_cof_dict[name] = cof
    # 读取需要预测的天数
    start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
    end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
    days = (end_time - start_time).days
    pred_result_dict = dict()
    for name in flavname:
        # 取数据的最后degree个样本点
        n_sum = 0  # flavor在该周期内的数量总和
        flavor_data = data_dict_filled[name]  # 数据点
        cofs = AR_cof_dict[name]  # 系数
        if len(flavor_data) < deg:
            print("%s" % name)
            pred_result_dict[name] = float(0)
            continue
        samples = flavor_data[-deg:]
        samples = [x[1] for x in samples]
        for day in range(days):
            x_pred = 0  # 预测值
            for i in range(len(cofs)):
                x_pred = x_pred + samples[i] * cofs[i]
            del samples[0]
            x_pred = math.floor(abs(x_pred))
            n_sum = n_sum + x_pred
            samples.append(x_pred)
        pred_result_dict[name] = n_sum
    print("预测完毕")

    # 放置服务器
    phy_server = dict()
    n_phy = 0
    idle_mem = 0
    idle_cpu = 0
    vm_type = mission.vm_type
    vm_cost = dict()
    for name in vm_type.keys():
        vm_cost[name] = int(vm_type[name][0]) * int(vm_type[name][1])
    vm_cost= sorted(vm_cost.items(), key=lambda x: x[1])
    for item in vm_cost:
        n_flavor = pred_result_dict[item[0]]
        for i in range(int(n_flavor)):
            n_mem = int(mission.vm_type[item[0]][1])
            n_cpu = int(mission.vm_type[item[0]][0])
            if n_mem <= idle_mem and n_cpu <= idle_cpu:
                idle_mem = idle_mem - n_mem
                idle_cpu = idle_cpu - n_cpu
                phy_server[n_phy].append(item[0])
            else:
                n_phy = n_phy + 1
                phy_server[n_phy] = []
                idle_mem = mission.phy_mem * 1024
                idle_cpu = mission.phy_cpu
                idle_mem = idle_mem - n_mem
                idle_cpu = idle_cpu - n_cpu
                phy_server[n_phy].append(item[0])

    print("放置完成")
    print("生成结果输出文件")
    n_sum = 0
    for name in pred_result_dict.keys():
        n_sum = n_sum + pred_result_dict[name]
    result.append(int(n_sum))
    for name in pred_result_dict.keys():
        result.append("%s %d" % (name, pred_result_dict[name]))
    result.append("")
    result.append(len(phy_server))
    for n in phy_server.keys():
        s = str(n)
        u_name = set(phy_server[n])
        for u_n in u_name:
            cnt = phy_server[n].count(u_n)
            s = s + ' ' + str(u_n) + ' ' + str(cnt)
        result.append(s)
    return result
