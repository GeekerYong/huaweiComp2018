# coding=utf-8
import preprocessing
import plot
import scipy.io as sio
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
    save_mat = [d[1] for d in data_dict_filled['flavor14']]
    sio.savemat('flavor14.mat', {'array': save_mat})
    plot.plot_flavor_data(data_dict_filled['flavor14'], 'flavor14')
    return result
