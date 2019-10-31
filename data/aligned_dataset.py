import os.path
import random
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch

class AlignedDataset(BaseDataset):
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
        self.transform_A = get_simple_transform(grayscale=(input_nc == 1))
        self.transform_B = get_simple_transform(grayscale=(output_nc == 1))

    def __getitem__(self, index):
        # / home / tushar / data / celeba_full/ train /celeba220_face0358_rec.png
        # /home/tushar/data2/GAN_CelebA.MoFA1/train/0142_recon_000002.png
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # make changes here
        '''
        list_path_b = AB_path.split('/')
        target_path = os.path.join('/',list_path_b[1], list_path_b[2], list_path_b[3], list_path_b[4], "target",
                                  list_path_b[6].split("_")[0],'orig'+'_'+list_path_b[6].split("_")[2][:-3]+'jpg')
        '''

        '''
        if self.opt.isTrain:
            target_path  = AB_path.replace('train','target').replace(AB_path.split('_')[-1],'76.png')
        else:
            target_path  = AB_path.replace('test','target').replace(AB_path.split('_')[-1],'76.png')

        
        target_path_normal = os.path.join('/', list_path_b[1], list_path_b[2], list_path_b[3], list_path_b[4], "target",
                                   list_path_b[6].split("_")[0],
                                   'norm' + '_' + list_path_b[6].split("_")[2][:-3] + 'png')

        target_path_shading = os.path.join('/', list_path_b[1], list_path_b[2], list_path_b[3], list_path_b[4], "target",
                                          list_path_b[6].split("_")[0],
                                          'shading' + '_' + list_path_b[6].split("_")[2][:-3] + 'png')

        target_path_geometry = os.path.join('/', list_path_b[1], list_path_b[2], list_path_b[3], list_path_b[4],
                                           "target",
                                           list_path_b[6].split("_")[0],
                                           'geometry' + '_' + list_path_b[6].split("_")[2][:-3] + 'png')
        '''


        if self.opt.isTrain:
            target_path = AB_path.replace('train','target')
        else:
            target_path = AB_path.replace('test','target')

        # print(AB_path)
        # print(target_path)

        w, h = AB.size
        w2 = int(w / 2)
        A0 = AB.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)


        # B_im = Image.open(target_path).convert('RGB')
        B_im = Image.open(target_path)
        if self.opt.isTrain:
            _,_,_,alpha = B_im.split()
        else:
            alpha = B_im
        B0 = alpha.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        # print(A0.size)
        # print(B0.size)


        # C_im = Image.open(target_path_normal).convert('RGB')
        # C0 = C_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        #
        # D_im = Image.open(target_path_shading).convert('RGB')
        # D0 = D_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        #
        # # E_im = Image.open(target_path_shading).convert('RGB')
        # # E0 = E_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        #
        # E_im = Image.open(target_path_geometry).convert('RGB')
        # E0 = E_im.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        x, y, h, w = transforms.RandomCrop.get_params(A0, output_size=[self.opt.fineSize, self.opt.fineSize])
        '''
        print(AB_path)
        print(target_path)
        print(target_path_normal)
        print(target_path_shading)
        print(target_path_geometry)
        import time
        time.sleep(400)
        '''
        A = TF.crop(A0, x, y, h, w)
        B = TF.crop(B0, x, y, h, w)
        # C = TF.crop(C0, x, y, h, w)
        # D = TF.crop(D0, x, y, h, w)
        # E = TF.crop(E0, x, y, h, w)

        # here make a load function for the npy files!
        # add shading and normal
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)
            # C = TF.hflip(C)
            # D = TF.hflip(D)
            # E = TF.hflip(E)
        A = self.transform_A(A)
        B = self.transform_B(B)
        # C = self.transform_A(C)
        # D = self.transform_A(D)
        # E = self.transform_A(E)

        # A_1  = torch.cat((A, C, D, E), 0)

        # import time
        # time.sleep(20)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'