import os.path
import random
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np
import torch
from torch.autograd import Variable
class lightDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.loadSize >= self.opt.fineSize)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_A = get_simple_transform(grayscale=False)
        self.transform_B = get_simple_transform(grayscale=False)

    def __getitem__(self, index):
        # todo: A,B,AL,BL
        AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')


        A = cv2.imread(AB_path)
        row, col= A.shape[0] , A.shape[1]
        img_A = cv2.resize(A, (512, 512))
        Lab_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2LAB)
        inputL = Lab_A[:, :, 0]
        inputA = inputL.astype(np.float32) / 255.0 #totensor also dividing????
        inputA = inputA[ ... ,None]


        # make changes here

        if self.opt.isTrain:
            target_path = AB_path.replace('train','target')
        else:
            target_path = AB_path.replace('test','target')

        # TODO: target path for now
        # target_path = '/home/tushar/DPR_data_light/light/target/imgHQ00000_01.png'
        B = cv2.imread(target_path)
        row, col = B.shape[0], B.shape[1]
        img_B = cv2.resize(B, (512, 512))
        Lab_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2LAB)
        inputL = Lab_B[:, :, 0]
        inputB = inputL.astype(np.float32) / 255.0
        inputB = inputB[ ... ,None]


        # TODO NORMALISE??????? Check base dataset
        A = self.transform_A(inputA)
        B = self.transform_B(inputB)


        # TODO: LIGHT!!!!!
        AL_path =AB_path.replace('train','train_light') + '_light.mat'
        # AL_path = '/home/tushar/DPR_data_light/light/train/imgHQ00000_light_00.txt'
        sh = np.loadtxt(AL_path)
        sh = sh[0:9]
        sh_AL = sh * 0.7
        sh_AL = np.squeeze(sh_AL)
        sh_AL = np.reshape(sh_AL, (9, 1, 1)).astype(np.float32)

        BL_path = target_path.replace('target','target_light') + '_light.mat'
        # BL_path = '/home/tushar/DPR_data_light/light/target/imgHQ00000_light_01.txt'
        sh = np.loadtxt(BL_path)
        sh = sh[0:9]
        sh_BL = sh * 0.7
        sh_BL = np.squeeze(sh_BL)
        sh_BL = np.reshape(sh_BL, (9, 1, 1)).astype(np.float32)

        # todo: check for just VARIABLE thingy

        return {'A': A, 'B': B, 'AL':torch.from_numpy(sh_AL),'BL':torch.from_numpy(sh_BL), 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'lightDataset'