import os
import numpy as np
from PIL import Image
import random
from skimage.transform import resize


class Dataloader:
    def __init__(self, opt):
        self.data_size = opt.Datasize
        self.data_num = opt.Datanum     #number of data
        self.dataset = opt.Dataset
        self.Depth = opt.FrameNum
        self.seed = opt.Seed
        self.type = opt.DataType

    def dataloader(self, root_dir):
        dataset = []
        labels = []
        class_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        for label, class_folder in enumerate(class_list):
            class_path = os.path.join(root_dir, class_folder)
            random.seed(self.seed)
            # train_path= random.sample(path_list, self.data_num)
            data = self._load_data(class_path)
            dataset.append(data)
            labels.append(label)

        Datasets = np.array(dataset)
        Datasets = np.moveaxis(Datasets, (1, 2), (0, -1))

        labels = np.stack(labels, axis=0)
        # labels = np.array(labels)

        return Datasets, labels, class_list

    def _load_data(self, set_path):
        file_list = sorted([filename for filename in os.listdir(set_path) if filename != "._DAV"]
                           , key=self.natural_sort_key)
        for img_filename in file_list:
            img_path = os.path.join(set_path, img_filename)
            img = Image.open(img_path).convert(self.type)
            img_array = np.array(img)
            img_array = np.moveaxis(img_array, -1, 0)
            if self.data_size != img_array.shape[-1]:
                img_array = resize(img_array, (img_array.shape[0], self.data_size, self.data_size))

        return img_array

    def natural_sort_key(self, s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def normalizing(self, img):
        min = np.min(img)
        max = np.max(img)
        scaled = (img - min)/(max - min)
        return scaled
