import os
import numpy as np
import torch
import random
from skimage.measure import block_reduce

def get_model_setting(opt):
    if opt.ModelName in ('PixelHop', 'PixelHop++', 'VoxelHop'):
        model_setting = (opt.ModelName + '_' + opt.Dataset + '_' + str(opt.NumUnit) + '_' + str(opt.Datasize)
                         + '_Thr' + str(opt.Th1) + '_' + str(opt.Th2)
                         + '_Seed' + str(opt.Seed) + '_LR' + str(opt.XGB_LR) + '_est' + str(opt.N_estimator)
                         + '_Depth' + str(opt.Max_Depth))
    # elif opt.ModelName == 'PixelHop++':
    #     model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset \
    #
    # elif opt.ModelName == 'VoxelHop':
    #     model_setting = opt.ModelName + '_'
    else:
        raise ValueError('*****Wrong Model Name*****')
    return model_setting


def seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


def get_subdir_list(path, is_sort=True):
    subdir_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if is_sort:
        subdir_list.sort()
    return subdir_list


def get_file_list(path, is_sort=True):
    file_list = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    if is_sort:
        file_list.sort()
    return file_list


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MaxPooling(x, modelname):
    if modelname == 'VoxelHop':
        return block_reduce(x, (1, 2, 2, 1, 1), np.max, cval=np.max)
    else:
        return block_reduce(x, (1, 2, 2, 1), np.max)


def AvgPooling(x, modelname):
    if modelname == 'VoxelHop':
        return block_reduce(x, (1, 2, 2, 1, 1), np.mean)
    else:
        return block_reduce(x, (1, 2, 2, 1), np.mean)
