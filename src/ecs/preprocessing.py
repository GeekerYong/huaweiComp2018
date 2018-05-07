# -*- coding:utf-8 -*- 
#  @Time : 2018-03-20 10:38
#  @Author : GeekerYong
#  @File : preprocessing.py
#  @Description:
#           This script contains the functions needed for preprocessing

import datetime


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
        if line != '\n':  # linux下为\r\n
            if cnt == 0:
                phy_cpu, phy_mem, phy_hard_disk = line.split(" ")
                # print("0:" + line)
            elif cnt == 1:
                vm_info.append(line.replace('\n', ''))
                # print("1:" + line)
            elif cnt == 2:
                opt_target = line.replace('\n', '')
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
    """
    flavor_dict = dict()
    for item in ecs_lines:
        values = item.split("\t")
        #print(values)
        vm_name = values[1]
        uuid = values[0]
        create_time = values[2].split(" ")[0]
        if vm_name in mission.vm_type.keys():
            if vm_name not in flavor_dict.keys():
                flavor_dict[vm_name] = [[uuid, create_time, vm_name]]
            else:
                flavor_dict[vm_name].append([uuid, create_time, vm_name])
        else:
            continue
        # print("用户id：%s 配置名称:%s 创建时间:%s" % (uuid, flavorName, createTime))
    print("ecs_data处理完毕")
    return flavor_dict


def merge(data_dict, mission):
    # 按flavor,合并同一日期的数据
    keys = data_dict.keys()
    data_dict_merge = dict()
    for key in keys:
        data_list = data_dict[key]
        last_day = data_list[0][1]
        flavor_num = 0
        for item in data_list:
            cur_day = item[1]
            if cur_day == last_day:
                flavor_num += 1
            else:
                if (key not in data_dict_merge.keys()):
                    data_dict_merge[key] = [[last_day, flavor_num]]
                else:
                    data_dict_merge[key].append([last_day, flavor_num])
                flavor_num = 1
            last_day = cur_day
        # 处理最后一个日期下的数据
        if (key not in data_dict_merge.keys()):
            data_dict_merge[key] = [[last_day, flavor_num]]
        else:
            data_dict_merge[key].append([last_day, flavor_num])
    print("分析完成")
    return data_dict_merge
    # if mission.opt_target == "CPU":
    #     return None
    # else:
    #     return None


def fill_data(merge_data, mission):
    for key in merge_data.keys():
        data = merge_data[key]
        new_data = []
        start_date = datetime.datetime.strptime(data[0][0], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(mission.time_limit[0].split(' ')[0], '%Y-%m-%d')
        date_list = [d[0] for d in data]
        date_list_comp = []
        for i in range((end_date-start_date).days):
            now_date = start_date + datetime.timedelta(days=i)
            date_list_comp.append(now_date.strftime('%Y-%m-%d'))
        for date in date_list_comp:
            if date in date_list:
                new_data.append(data[date_list.index(date)])
            else:
                new_data.append([date, 0])
        merge_data[key] = new_data
    print("生成初始日期")
    return merge_data
