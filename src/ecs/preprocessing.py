# -*- coding:utf-8 -*- 
#  @Time : 2018-03-20 10:38
#  @Author : Liu JinYong
#  @File : preprocessing.py
#  @Description:
#           This script contains the functions needed for preprocessing


class Mission:

    def __init__(self, phy_cpu, phy_mem, phy_hard_disk, vm_type, opt_target, time_limit):
        self.phy_cpu = phy_cpu
        self.phy_mem = phy_mem
        self.phy_hard_disk = phy_hard_disk
        self.vm_type = vm_type
        self.opt_target = opt_target
        self.time_limit = time_limit


def preprocess_input(input_lines):
    """
        处理input文件
    :param input_lines: 从文件中读取的以字符串形式存储的input信息
    :return:
            mission：分析input文件得到的任务对象，包含优化任务所需的所有信息
    """
    cnt = 0
    vm_info = []
    time_limit = []
    for line in input_lines:
        if line != '\n':
            if cnt == 0:
                phy_cpu, phy_mem, phy_hard_disk = line.split(" ")
                # print("0:" + line)
            elif cnt == 1:
                vm_info.append(line.replace('\n', ''))
                # print("1:" + line)
            elif cnt == 2:
                opt_target = line
                # print("2:" + line)
            elif cnt == 3:
                time_limit.append(line.replace('\n', ''))
                # print("3:" + line)
        else:
            cnt += 1
            continue
    vm_type = dict()
    for i in range(len(vm_info)):
        if i == 0:
            continue
        else:
            info = []
            values = vm_info[i].split(" ")
            vm_name = values[0]
            info.append(values[1])
            info.append(values[2])
            vm_type[vm_name] = info
    mission = Mission(int(phy_cpu), int(phy_mem), int(phy_hard_disk), vm_type, opt_target, time_limit)
    print("input_data处理完毕")
    return mission


def preprocess_ecs_info(ecs_lines, mission):
    """
        处理ecs文件
    :param ecs_lines: 从文件中读取的以字符串形式存储的ecs历史信息
    :param mission: 分析input文件得到的任务对象，包含优化任务所需的所有信息
    :return:
            所有返回数据已剔除不相关的历史数据
            flavor_dict：以vm_name为键，键对应内容为list格式，包含所有该vm_name的历史请求数据。
            ecs_data:以list存储的历史请求数据。
    """
    flavor_dict = dict.fromkeys(mission.vm_type.keys(), [])
    ecs_data = []
    for item in ecs_lines:
        values = item.split("\t")
        vm_name = values[1]
        if vm_name in flavor_dict.keys():
            # 生成dict
            info = []
            uuid = values[0]
            create_time = values[2].split(" ")[0]
            info.append(uuid)
            info.append(create_time)
            flavor_dict[vm_name].append(info)
            # 普通组织形式
            info.append(vm_name)
            ecs_data.append(info)
        else:
            continue
        # print("用户id：%s 配置名称:%s 创建时间:%s" % (uuid, flavorName, createTime))
    print("ecs_data处理完毕")
    return flavor_dict,ecs_data
