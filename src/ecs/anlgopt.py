# -*- coding:utf-8 -*- 
#  @Time : 2018-04-13 9:58
#  @Author : GeekerYong
#  @File : anlgopt.py
#  @Description:
#            模拟退火
#   --Input:
#   --Output:
import random
import copy
import math
from operator import itemgetter
import time

def annealingoptimize(pred_result_dict, mission, T=100000.0, cool=0.99, step=1):

    pred_result = [[i, pred_result_dict[i],0] for i in pred_result_dict.keys()]
    for item in pred_result:
        name = item[0]
        n_mem = int(mission.vm_type[name][1])
        n_cpu = int(mission.vm_type[name][0])
        cost = item[1]*n_mem*n_cpu*0.00001
        item[2] = cost
    pred_result_list = sorted(pred_result, key=itemgetter(2))
    best_phy_server = dict()
    best_cost = 99999999
    cnt = 0
    start = time.time()

    while (time.time() - start)<15:
        vec = [random.randint(0, len(pred_result_list)-1) for i in range(len(pred_result_list))]  # 骰子,会更换前两个数代表的服务器
        if vec[0] != vec[1]:
            pred_result_list[vec[0]],pred_result_list[vec[1]] = pred_result_list[vec[1]],pred_result_list[vec[0]] # 被选中的服务器进行交换
        else:
            continue
        phy_server = dict()
        phy_server_usage = dict()
        n_phy = 0
        idle_mem = 0
        idle_cpu = 0
        for item in pred_result_list:
            n_flavor = item[1]
            for i in range(int(n_flavor)):
                flag = 0 # 放置flag
                n_mem = int(mission.vm_type[item[0]][1])
                n_cpu = int(mission.vm_type[item[0]][0])
                for server in phy_server_usage.keys(): # 遍历当前所有服务器
                    usage = phy_server_usage[server]
                    idle_cpu = usage[0]
                    idle_mem = usage[1]
                    if n_mem <= idle_mem and n_cpu <= idle_cpu:
                        idle_cpu = idle_cpu - n_cpu
                        idle_mem = idle_mem - n_mem
                        usage_update = [idle_cpu,idle_mem]
                        phy_server_usage[server] = usage_update # 更新空闲空间
                        phy_server[server].append(item[0]) # 放置进去
                        flag = 1
                        break
                if flag==0:
                    n_phy = n_phy + 1
                    phy_server[n_phy] = []
                    idle_mem = mission.phy_mem * 1024
                    idle_cpu = mission.phy_cpu
                    idle_mem = idle_mem - n_mem
                    idle_cpu = idle_cpu - n_cpu
                    usage = [idle_cpu, idle_mem]
                    phy_server_usage[n_phy] = usage
                    phy_server[n_phy].append(item[0])

                # if n_mem <= idle_mem and n_cpu <= idle_cpu:
                #     idle_mem = idle_mem - n_mem
                #     idle_cpu = idle_cpu - n_cpu
                #     phy_server[n_phy].append(item[0])
                # else:
                #     n_phy = n_phy + 1
                #     phy_server[n_phy] = []
                #     idle_mem = mission.phy_mem * 1024
                #     idle_cpu = mission.phy_cpu
                #     idle_mem = idle_mem - n_mem
                #     idle_cpu = idle_cpu - n_cpu
                #     usage = [idle_mem,idle_cpu]
                #     phy_server[n_phy].append(item[0])
        cost = 0
        for server in phy_server_usage.keys():
            usage = phy_server_usage[server]
            cost =cost + usage[0]/mission.phy_cpu + usage[1]/mission.phy_mem
        cost = cost * n_phy
        #cost = n_phy * mission.phy_mem * mission.phy_cpu

        print("当前最好成本: %d" %best_cost)
        print("当前成本: %d" % cost)
        if cost < best_cost or random.random() <math.exp(-(cost-best_cost)/T):
            best_phy_server = copy.deepcopy(phy_server)
            best_cost = cost
        T = T*cool
    print("模拟退火运算完毕")
    return phy_server


if __name__ == '__main__':
    vm_cost = [('flavor1', 10), ]
    annealingoptimize()
