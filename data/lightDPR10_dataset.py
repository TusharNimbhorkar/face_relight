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
class lightDPR10Dataset(BaseDataset):
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
        for i in range(0, len(self.AB_paths_) - 6, 6):
            i1 = self.AB_paths_[i:i + 6]
            i2 = self.AB_paths_[i:i + 5]

            self.list_AB.append([i1[-1], i1[-1]])
            self.AB_paths.append(i1[-1])

            syn1 = random.sample(i2,2)
            for s1 in syn1:
                self.list_AB.append([s1, i1[-1]])
                self.AB_paths.append(i1[-1])


            i1 = self.AB_paths_[i:i + 5]
            i2 = self.AB_paths_[i:i + 5]
            for k in range(5):
                a = [i1.pop(random.randrange(len(i1))) for _ in range(1)]
                blist = list(set(i2) - set(a))
                b = [blist.pop(random.randrange(len(blist))) for _ in range(1)]
                blist.append(a[0])
                i2 = blist
                self.AB_paths.append(a[0])
                self.list_AB.append([a[0] , b[0]])

            # i1 = self.AB_paths_[i:i + 5]
            # i2 = self.AB_paths_[i:i + 5]
            #
            # for l in range(2):
            #     a = [i1.pop(random.randrange(len(i1))) for _ in range(1)]
            #     blist = list(set(i2) - set(a))
            #     b = [blist.pop(random.randrange(len(blist))) for _ in range(1)]
            #     blist.append(a[0])
            #     i2 = blist
            #     self.AB_paths.append(a[0])
            #     self.list_AB.append([a[0] , b[0]])


        random.shuffle(self.list_AB)

        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.loadSize >= self.opt.fineSize)

        self.transform_A = get_simple_transform(grayscale=False)
    def __getitem__(self, index):
        # todo: A,B,AL,BL

        real_im_number = random.choice(range(0, self.opt.ffhq))

        real_im_path = os.path.join(self.opt.dataroot,'real_im',"{:05d}".format(real_im_number)+'.png')
        C = cv2.imread(real_im_path)
        img_C = cv2.resize(C, (512, 512))
        Lab_C = cv2.cvtColor(img_C, cv2.COLOR_BGR2LAB)
        inputLC = Lab_C[:, :, 0]
        inputC = inputLC.astype(np.float32) / 255.0
        inputC = inputC.transpose((0,1))
        inputC = inputC[..., None]


        AB_path = self.list_AB[index]
        A = cv2.imread(AB_path[0])
        img_A = cv2.resize(A, (512, 512))
        Lab_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2LAB)
        inputLA = Lab_A[:, :, 0]
        inputA = inputLA.astype(np.float32) / 255.0  #totensor also dividing????
        inputA = inputA.transpose((0, 1))
        inputA = inputA[ ... ,None]

        if self.opt.isTrain:
            target_path = AB_path[1]
        else:
            # TODO: CHANGE THIS IN FUTURE
            target_path = AB_path.replace('test','target')

        B = cv2.imread(target_path)
        img_B = cv2.resize(B, (512, 512))
        Lab_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2LAB)
        inputLB = Lab_B[:, :, 0]
        inputB = inputLB.astype(np.float32) / 255.0
        inputB = inputB.transpose((0, 1))
        inputB = inputB[ ... ,None]

        '''
        orig_im_path = os.path.join(self.opt.dataroot, 'orig',
                                    "{:05d}".format(int(AB_path.split('/')[-1].split('_')[0][5:]) + 1) + '.jpg')

        orig = cv2.imread(orig_im_path)
        img_orig = cv2.resize(orig, (512, 512))
        Lab_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2LAB)
        inputLorig = Lab_orig[:, :, 0]
        inputorig = inputLorig.astype(np.float32) / 255.0
        inputorig = inputorig.transpose((0, 1))
        inputorig = inputorig[..., None]
        '''


        # TODO NORMALISE??????? Check base dataset
        A = self.transform_A(inputA)
        B = self.transform_A(inputB)
        C = self.transform_A(inputC)
        # D = self.transform_A(inputorig)

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
        sh_BL = sh * 1.0 #0.7
        sh_BL = np.squeeze(sh_BL)
        sh_BL = np.reshape(sh_BL, (9, 1, 1)).astype(np.float32)

        # todo: check for just VARIABLE thingy

        return {'A': A, 'B': B,'C':C, 'AL':torch.from_numpy(sh_AL),'BL':torch.from_numpy(sh_BL), 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.list_AB)

    def name(self):
        return 'lightDPR10Dataset'
