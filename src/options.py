import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--UseCUDA', help='Use CUDA?', type=str2bool, nargs='?', default=False)
        parser.add_argument('--NumWorker', help='num of worker for dataloader', type=int, default=1)
        parser.add_argument('--Mode', help='script mode', choices=['Train', 'Test'], default='Train')
        parser.add_argument('--ModelName', help='PixelHop/VoxelHop', type=str, default='VoxelHop')
        parser.add_argument('--NumUnit', help='num of unit for network', type=int, default=5)
        parser.add_argument('--Seed', type=int, default=10)
        parser.add_argument('--Dataset', help='Dataset', type=str, default='/TBR_easy')
        parser.add_argument('--DataType', help='Type of Data, Voxel/Event-Voxel/Gray/RGB', type=str, default='Voxel')
        parser.add_argument('--Datasize', help='Image or Voxel W/H', type=int, default=128)
        parser.add_argument('--Datanum', help='Number of data', type=int, default=20)
        parser.add_argument('--ImgChnNum', help='image channel', type=int, default=2)
        parser.add_argument('--FrameNum', help='Frame length - Voxel length', type=int, default=5)
        #### Saab Transform
        parser.add_argument('--UseDC', help='UseDC or not', type=float, default=True)
        parser.add_argument('--pad', help='reflect/zeros/none', type=str, default='reflect')
        parser.add_argument('--dilate', help='dilate', type=float, default=1)
        parser.add_argument('--Th1', help='whole PCA threshold', type=float, default=0.99)
        parser.add_argument('--Th2', help='each PCA threshold', type=float, default=0.005)
        # ##
        parser.add_argument('--DataRoot', help='DataPath', type=str, default='./Dataset')
        parser.add_argument('--ModelRoot', help='Path for saving model', type=str, default='./models/')
        ####LAG
        parser.add_argument('--num_cluster', help='output feature shape', type=float, default=200)
        parser.add_argument('--alpha', help='alpha', type=float, default=10)
        #### XGBoost Classifier
        parser.add_argument('--XGB_LR', help='learning rate', type=float, default=1e-4)
        parser.add_argument('--N_estimator', help='learning rate', type=float, default=1e-4)
        parser.add_argument('--Max_Depth', help='learning rate', type=float, default=1e-4)

        self.initialized = True
        self.parser = parser
        return parser

    def print_options(self, opt):
        # This function is adapted from 'cycleGAN' project.
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def parse(self, is_print):
        parser = self.initialize()
        opt = parser.parse_args()
        if(is_print):
            self.print_options(opt)
        self.opt = opt
        return self.opt



