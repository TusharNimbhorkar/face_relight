import os.path
import random
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch


class tripletdataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        assert (opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert (self.opt.loadSize >= self.opt.fineSize)
        input_nc = 3
        output_nc = 3
        self.transform_A = get_simple_transform(grayscale=(input_nc == 1))
        self.transform_B = get_simple_transform(grayscale=(output_nc == 1))
        self.transform_C = get_simple_transform(grayscale=(output_nc == 1))

    def __getitem__(self, index):

        ABC_path = self.ABC_paths[index]
        # 0-71 -> 4, 72-143 -> 76, 144-215 -> 148
        if self.opt.isTrain:
            B_path = ABC_path.replace('train', 'target')

            if int(ABC_path.split('_')[-1].split('.')[0]) <= 71:
                C_path = ABC_path.replace('train', 'train').replace(ABC_path.split('_')[-1], '4.png')
            elif int(ABC_path.split('_')[-1].split('.')[0]) > 71 and int(ABC_path.split('_')[-1].split('.')[0]) <=143:
                C_path = ABC_path.replace('train', 'train').replace(ABC_path.split('_')[-1], '76.png')

            else:
                C_path = ABC_path.replace('train', 'train').replace(ABC_path.split('_')[-1], '148.png')
            #C_path = ABC_path.replace('train', 'train').replace(ABC_path.split('_')[-1], '76.png')
        else:
            B_path = ABC_path.replace('test', 'target')

            if int(ABC_path.split('_')[-1].split('.')[0]) <= 71:
                C_path = ABC_path.replace('test', 'test').replace(ABC_path.split('_')[-1], '4.png')
            elif int(ABC_path.split('_')[-1].split('.')[0]) > 71 and int(ABC_path.split('_')[-1].split('.')[0]) <=143:
                C_path = ABC_path.replace('test', 'test').replace(ABC_path.split('_')[-1], '76.png')

            else:
                C_path = ABC_path.replace('test', 'test').replace(ABC_path.split('_')[-1], '148.png')
            #C_path = ABC_path.replace('test', 'test').replace(ABC_path.split('_')[-1], '76.png')

        A_im = Image.open(ABC_path).convert('RGB')

        B_im = Image.open(B_path)
        if self.opt.isTrain:
            _, _, _, alpha = B_im.split()
        else:
            alpha = B_im
        C_im = Image.open(C_path).convert('RGB')

        A0 = A_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B0 = alpha.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        C0 = C_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        x, y, h, w = transforms.RandomCrop.get_params(A0, output_size=[self.opt.fineSize, self.opt.fineSize])

        A = TF.crop(A0, x, y, h, w)
        B = TF.crop(B0, x, y, h, w)
        C = TF.crop(C0, x, y, h, w)

        if (not self.opt.no_flip) and random.random() < 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)
            C = TF.hflip(C)

        A = self.transform_A(A)
        B = self.transform_B(B)
        C = self.transform_C(C)

        return {'A': A, 'B': B, 'C': C, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'tripletdataset'
