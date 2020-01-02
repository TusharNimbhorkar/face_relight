# TRAIN ON SH ONLY
import torch
import sys
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
sys.path.append('.')
from skeleton512 import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class lightgrad58Model(BaseModel):
    def name(self):
        return 'lightgrad58Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_MSE']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.netG = HourglassNet().cuda()
        self.netG = torch.nn.DataParallel(self.netG, self.opt.gpu_ids)
        self.netG.train(True)

        if self.isTrain:
            self.mseloss = torch.nn.MSELoss(reduction='sum').to(self.device)

            # initialize optimizers
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters())

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        # self.real_B = input['B'].to(self.device)
        self.real_AL = input['AL'].to(self.device)
        self.real_BL = input['BL'].to(self.device)
        # self.real_C = input['C'].to(self.device)
        # self.real_D = input['D'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, epoch):

        count_skip = 4
        if epoch > 5:
            count_skip = 9 % epoch
        if epoch >= 10:
            count_skip = 0

        _, _, self.fake_AL, _ = self.netG(self.real_A, self.real_BL, count_skip, oriImg=None)

    def calc_gradient(self, x):
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).cuda())
        G_x = conv1(Variable(x)).data.view(self.opt.batch_size, 1, 512, 512)
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).cuda())
        G_y = conv2(Variable(x)).data.view(self.opt.batch_size, 1, 512, 512)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        return G

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_C, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_C, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self, epoch):

        self.loss_G_MSE = self.mseloss(self.real_AL, self.fake_AL)
        self.loss_L1_add = self.loss_G_MSE
        self.loss_G = self.loss_L1_add #self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward(epoch)
        # update G
        # self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
