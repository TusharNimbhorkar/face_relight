# Our dataloader.

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
import matplotlib.pyplot as plt
from commons.common_tools import Logger, BColors
from copy import deepcopy

log = Logger("DataLoader", tag_color=BColors.LightBlue)

class lightDPR7Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--ffhq', type=int, default=70000, help='sample size ffhq')

        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths_ = make_dataset(self.dir_AB)
        self.AB_paths = []
        self.dict_AB = {}
        self.list_AB = []

        self.img_size = self.opt.img_size
        self.use_segments = self.opt.segment
        synth_n_per_id = self.opt.n_synth
        n_want = self.opt.n_first
        n_per_id = synth_n_per_id + 1

        for i in range(0, len(self.AB_paths_) - n_per_id, n_per_id):

            ori_img = self.AB_paths_[i+synth_n_per_id]
            self.list_AB.append([ori_img, ori_img])

            i1 = self.AB_paths_[i:i + n_want]
            i1.append(ori_img)
            i2 = deepcopy(i1)

            self.AB_paths.append(i1[-1])
            for k in range(len(i1)):
                a = [i1.pop(random.randrange(len(i1))) for _ in range(1)]
                blist = list(set(i2) - set(a))
                b = [blist.pop(random.randrange(len(blist))) for _ in range(1)]
                # blist_final = list(set(blist).add(set(a)))
                blist.append(a[0])
                i2 = blist
                self.AB_paths.append(a[0])
                self.list_AB.append([a[0] , b[0]])
        random.shuffle(self.list_AB)

        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.loadSize >= self.opt.fineSize)

        self.transform_A = get_simple_transform(grayscale=False)

    def _get_paths(self, real_img_id, input_img_id):
        AB_path = self.list_AB[input_img_id]
        source_path = AB_path[0]
        target_path = AB_path[1]

        src_light_path = AB_path[0][:-6] + 'light_%s.txt' % AB_path[0].split('_')[-1][:-4]
        tgt_light_path = AB_path[0][:-6] + 'light_%s.txt' % target_path.split('_')[-1][:-4]

        real_img_path = os.path.join(self.opt.dataroot,'real_im',"{:05d}".format(real_img_id)+'.png')
        orig_img_path = target_path[:-5]+'5.png'
        segment_img_path = os.path.join(self.opt.dataroot, 'segments', AB_path[0].split('/')[-1].split('_')[0],
                     source_path.split('/')[-1].split('_')[0] + '.png')
        return real_img_path, orig_img_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path

    def _img_to_input(self, img):
        '''
        Converts a numpy RGB image array into an L channel input array
        :param img: RGB image numpy array
        :return: L channel input array
        '''

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        input_L = img_lab[:, :, 0]
        input_L_norm = input_L.astype(np.float32) / 255.0
        input_L_norm = input_L_norm.transpose((0,1))
        input_L_norm = input_L_norm[..., None]
        input = self.transform_A(input_L_norm)

        return input

    def _get_sh(self, sh):
        sh = sh[0:9]
        sh = sh * 1.0
        sh = np.squeeze(sh)
        sh = np.reshape(sh, (9, 1, 1)).astype(np.float32)
        return sh

    def _read_img(self, img_path, size):
        img = cv2.imread(img_path)
        try:
            if size < img.shape[0]:
                img = cv2.resize(img, (self.img_size, self.img_size))
        except Exception as e:
            print(img_path)
            raise e

        return img

    def __getitem__(self, index):
        real_img_id = random.choice(range(0, self.opt.ffhq))
        AB_path = self.list_AB[index]
        real_path, orig_path, source_path, target_path, segment_img_path, src_light_path, tgt_light_path = self._get_paths(real_img_id, index)

        #Real discriminator image
        img_real = cv2.imread(real_path)

        try:
            if self.img_size < img_real.shape[0]:
                    img_real = cv2.resize(img_real, (self.img_size, self.img_size))
        except Exception as e:
            print(real_path)
            raise e

        input_real = self._img_to_input(img_real)

        #Real original image
        img_orig = self._read_img(orig_path, self.img_size)
        input_orig = self._img_to_input(img_orig)

        #Segments
        back_original = None
        if self.use_segments:
            segment_im = cv2.imread(segment_img_path)
            segment_im=cv2.resize(segment_im,(self.img_size,self.img_size))

            segment_im[segment_im==255]=1
            segment_im_invert = np.invert(segment_im)
            segment_im_invert[segment_im_invert == 254] = 0
            segment_im_invert[segment_im_invert == 255] = 1

            back_original = np.multiply(img_orig, segment_im_invert)

        #Source image
        img_src = self._read_img(source_path, self.img_size)

        if self.use_segments:
            img_src = back_original + np.multiply(img_src, segment_im)

        input_src = self._img_to_input(img_src)

        #Target image
        img_tgt = self._read_img(target_path, self.img_size)

        if self.use_segments:
            img_tgt = back_original + np.multiply(img_tgt, segment_im)

        input_tgt = self._img_to_input(img_tgt)

        #Source light
        sh = np.loadtxt(src_light_path)
        sh_src = self._get_sh(sh)

        #Target light
        sh = np.loadtxt(tgt_light_path)
        sh_tgt = self._get_sh(sh)

        # todo: check for just VARIABLE thingy

        return {'A': input_src, 'B': input_tgt,'C':input_real,'D':input_orig, 'AL':torch.from_numpy(sh_src),'BL':torch.from_numpy(sh_tgt), 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.list_AB)

    def name(self):
        return 'lightDPR7Dataset'
