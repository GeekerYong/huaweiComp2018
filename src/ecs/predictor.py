# coding=utf-8
import preprocessing
import os
import datetime
import math
import subprocess
from matrix import *
from nn import *
from anlgopt import *

def generate_samples(samples, p):
    mat_x = matrix([[]])
    mat_x.zero(len(samples) - p, p)
    k = 0
    for i in range(0, len(samples) - p):
        for j in range(p):
            mat_x.value[i][j] = samples[k + j]
        k = k + 1
    mat_y = matrix([samples[p:len(samples)]])
    mat_y = mat_y.transpose()
    return mat_x, mat_y


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
    error_point = ['']
    if len(mission.vm_type) == 3:
        if mission.opt_target =="CPU":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 53
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=405)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)

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
        elif mission.opt_target=="MEM":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 55
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=40)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)

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
    elif len(mission.vm_type)==5:
        if mission.opt_target =="CPU":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 46
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=405)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)

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
        elif mission.opt_target=="MEM":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 60
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=40)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)

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
    else:
        if mission.opt_target =="CPU":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 42
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=405)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)

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
        elif mission.opt_target=="MEM":
            flavname = data_dict_filled.keys()
            nn_parameter_dict = dict()
            deg = 48
            for name in flavname:
                data = data_dict_filled[name]
                series = [x[1] for x in data]
                X, Y = generate_samples(series, deg)
                X = X.transpose()
                Y = Y.transpose()
                parameters = nn_model(X,Y,4,learning_rate=0.01,num_iterations=40)
                nn_parameter_dict[name] = parameters

            # 读取需要预测的天数
            start_time = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
            end_time = datetime.datetime.strptime(mission.time_limit[1].split(' ')[0], '%Y-%m-%d')
            days = (end_time - start_time).days

            pred_result_dict = dict()
            pred_list = []
            for name in flavname:
                # 取数据的最后degree个样本点
                n_sum = 0  # flavor在该周期内的数量总和
                flavor_data = data_dict_filled[name]  # 数据点
                parameters = nn_parameter_dict[name]  # 系数

                samples = flavor_data[-deg:]
                samples = [x[1] for x in samples]
                samples = matrix([samples])
                samples = samples.transpose()
                temp = []
                for day in range(days):
                    A2, cache = forward_propagation(samples,parameters)
                    x_pred = A2.value[0][0]
                    x_pred = math.ceil(abs(x_pred))
                    n_sum = n_sum + x_pred
                    samples.value[0][0] = x_pred
                    temp.append(x_pred)
                pred_result_dict[name] = n_sum
            print("预测完毕")

            # 放置服务器
            phy_server = annealingoptimize(pred_result_dict,mission,T=10000.0, cool=0.98)
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
