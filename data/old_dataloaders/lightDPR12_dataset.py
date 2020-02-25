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
from itertools import cycle
from torch.autograd import Variable
from pathlib import Path

class lightDPR12Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--ffhq', type=int, default=70000, help='sample size ffhq')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths_ = sorted(make_dataset(self.dir_AB))
        self.AB_paths = []
        self.dict_AB = {}
        self.list_AB = []
        self.list_SR = []
        self.list_RR = []
        self.valid_id = []
        with open('splits/train.lst') as f:
            lines = f.read().splitlines()

        for line in lines:
            self.list_AB.append([os.path.join(opt.dataroot, 'train', line.split(' ')[0], line.split(' ')[1]),
                                 os.path.join(opt.dataroot, 'train', line.split(' ')[0], line.split(' ')[2])])
            self.valid_id.append(line.split(' ')[0])
            self.AB_paths.append(line.split(' ')[1])

        self.valid_id = list(set(self.valid_id))
        for i in range(0, len(self.AB_paths_) - 6, 6):
            if self.AB_paths_[i:i + 6][0].split('/')[-2] in self.valid_id:

                i1 = self.AB_paths_[i:i + 6]
                i2 = self.AB_paths_[i:i + 6]

                self.list_RR.append([i1[-1], i1[-1]])
                self.AB_paths.append(i1[-1])
                i1 = self.AB_paths_[i:i + 5]
                self.list_SR.append([random.sample(i1, 1)[0], i2[-1]])
                self.AB_paths.append(i1[-1])

        ffhq_dir = os.path.join(self.opt.dataroot, 'real_im')
        self.list_ffhq = []
        for filename_im in Path(ffhq_dir).rglob('*.png'):
            self.list_ffhq.append(filename_im)



        random.shuffle(self.list_AB)
        random.shuffle(self.list_RR)
        random.shuffle(self.list_SR)
        random.shuffle(self.list_ffhq)

        self.syn_syn_it = cycle(self.list_AB)
        self.syn_ori_it = cycle(self.list_SR)
        self.ori_ori_it = cycle(self.list_RR)
        self.ffhq_it = cycle(self.list_ffhq)

        assert (opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert (self.opt.loadSize >= self.opt.fineSize)

        self.transform_A = get_simple_transform(grayscale=False)

    def __getitem__(self, index):
        # todo: A,B,AL,BL

        # choose set

        set_id = np.random.choice([0, 1, 2], 1, p=[0.05, 0.20, 0.75])
        if set_id == 0:
            sample = next(self.ori_ori_it)
        if set_id == 1:
            sample = next(self.syn_ori_it)
        if set_id == 2:
            sample = next(self.syn_syn_it)

        real_im_path = str(next(self.ffhq_it))
        C = cv2.imread(real_im_path)
        img_C = cv2.resize(C, (512, 512))
        Lab_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2LAB)
        inputLC = Lab_C[:, :, 0]
        inputC = inputLC.astype(np.float32) / 255.0
        inputC = inputC.transpose((0, 1))
        inputC = inputC[..., None]

        AB_path = sample
        A = cv2.imread(AB_path[0])
        img_A = cv2.resize(A, (512, 512))
        Lab_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2LAB)
        inputLA = Lab_A[:, :, 0]
        inputA = inputLA.astype(np.float32) / 255.0  # totensor also dividing????
        inputA = inputA.transpose((0, 1))
        inputA = inputA[..., None]

        if self.opt.isTrain:
            target_path = AB_path[1]
        else:
            # TODO: CHANGE THIS IN FUTURE
            target_path = AB_path.replace('test', 'target')

        B = cv2.imread(target_path)
        img_B = cv2.resize(B, (512, 512))
        Lab_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2LAB)
        inputLB = Lab_B[:, :, 0]
        inputB = inputLB.astype(np.float32) / 255.0
        inputB = inputB.transpose((0, 1))
        inputB = inputB[..., None]

        orig_im_path = sample[0][:-5] + '5.png'

        orig = cv2.imread(orig_im_path)
        img_orig = cv2.resize(orig, (512, 512))
        Lab_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2LAB)
        inputLorig = Lab_orig[:, :, 0]
        inputorig = inputLorig.astype(np.float32) / 255.0
        inputorig = inputorig.transpose((0, 1))
        inputorig = inputorig[..., None]

        # TODO NORMALISE??????? Check base dataset
        A = self.transform_A(inputA)
        B = self.transform_A(inputB)
        C = self.transform_A(inputC)
        D = self.transform_A(inputorig)

        del_item = AB_path[0].split('_')[-1][:-4]
        target_item = target_path.split('_')[-1][:-4]

        # TODO: LIGHT!!!!!
        AL_path = AB_path[0][:-6] + 'light_' + del_item + '.txt'
        sh = np.loadtxt(AL_path)
        sh = sh[0:9]
        sh_AL = sh * 1.0
        sh_AL = np.squeeze(sh_AL)
        sh_AL = np.reshape(sh_AL, (9, 1, 1)).astype(np.float32)

        BL_path = AB_path[0][:-6] + 'light_' + target_item + '.txt'
        sh = np.loadtxt(BL_path)
        sh = sh[0:9]
        sh_BL = sh * 1.0  # 0.7
        sh_BL = np.squeeze(sh_BL)
        sh_BL = np.reshape(sh_BL, (9, 1, 1)).astype(np.float32)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'AL': torch.from_numpy(sh_AL), 'BL': torch.from_numpy(sh_BL),
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'lightDPR12Dataset'
