import numpy as np

from src.options import Options
import os
from src.Utils import utils, VoxelLoader, FrameLoader
from src.models import PixelHop, VoxelHop, LAG_UNIT

opt_parser = Options()
opt = opt_parser.parse(is_print=True)

utils.seed(opt.Seed)
model_setting = utils.get_model_setting(opt)
print('Setting: %s' % model_setting)

saving_model_path = os.path.join(opt.ModelRoot, 'model_' + model_setting + '/')
utils.mkdir(saving_model_path)
data_root = opt.DataRoot + '/' + opt.Dataset + '/'

if opt.ModelName == 'PixelHop++':
    Frame_Loader = FrameLoader.Dataloader(opt)
    (Data, labels, class_list) = Frame_Loader.dataloader(data_root)
    model = PixelHop(opt, saving_model_path)
elif opt.ModelName == 'VoxelHop':
    Voxel_loader = VoxelLoader.Dataloader(opt)
    (Data, labels, class_list) = Voxel_loader.dataloader(data_root)
    model = VoxelHop(opt, saving_model_path)
else:
    raise ValueError('*****Wrong Model Name*****')

LAG = LAG_UNIT(opt, train_labels=labels, class_list=class_list)


if opt.Mode == 'Train':
    for i in range(opt.NumUnit):
        S = list(Data.shape)
        S0 = S[0]
        S[0] = 1
        feature_of_channel_output = []
        for j in range(S0):
            print(Data.shape)
            INPUT = Data[j].reshape(S)
            INPUT = np.moveaxis(INPUT, 0, -1)
            feature = model(i + 1, INPUT)

            # Intermediate, Leaf = unit.Unit(input)
            feature = utils.MaxPooling(feature, opt.ModelName)
            feature_of_channel_output.append(np.moveaxis(feature, -1, 0))

            feature = utils.AvgPooling(feature, opt.ModelName).reshape(feature.shape[0], -1)
            # feature = LAG.LAG_Unit(feature, Mode=opt.Mode, Unit_num=i+1, Channel_num=j+1)
        Data = np.concatenate(feature_of_channel_output, axis=0)
else:
    for i in range(opt.NumUnit):
        for j in range(len(Data.shape[-1])):
            feature = model(Data[:,:,:,:,j])
            # Intermediate, Leaf = unit.Unit(input)
            feature = utils.MaxPooling(feature, opt.ModelName)
            Data = feature
            # Intermediate = MaxPool_3D(Intermediate)
            feature = utils.AvgPooling(feature, opt.ModelName).reshape(feature.shape[0], -1)
            feature = LAG.LAG_Unit(feature, Mode=opt.Mode, Unit_num=i+1, Channel_num=j+1)