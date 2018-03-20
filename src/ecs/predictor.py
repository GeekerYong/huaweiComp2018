# coding=utf-8
import preprocessing
import plot

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
    plot.plot_flavor_data(data_dict_merge['flavor12'], 'flavor14')
    return result
