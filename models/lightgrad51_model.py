import torch
import sys
from .base_model import BaseModel
from . import networks
sys.path.append('.')
from relight_model import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class lightgrad51Model(BaseModel):
    def name(self):
        return 'lightgrad51Model'

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
        # self.loss_names = ['G_GAN', 'G_L1','G_MSE','G_total_variance', 'D_real', 'D_fake']
        self.loss_names = ['G_L1','G_MSE', 'G_total_variance']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.netG = HourglassNet().cuda()
        networks.init_weights(self.netG)


        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss(reduction='sum')

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters())

            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_AL = input['AL'].to(self.device)
        self.real_BL = input['BL'].to(self.device)
        self.real_C = input['C'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self,epoch):

        count_skip = 4
        if epoch > 5:
            count_skip = 9 % epoch
        if epoch > 10:
            count_skip = 0

        self.fake_B, self.fake_AL, _ = self.netG(self.real_A,self.real_BL,count_skip)

    def calc_gradient(self,x):

        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).cuda())
        G_x = conv1(Variable(x)).data.view(1, 512, 512)
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).cuda())
        G_y = conv2(Variable(x)).data.view(1, 512, 512)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        return G


    def backward_G(self,epoch):

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B)

        self.loss_G_MSE = self.mseloss(self.real_AL, self.fake_AL)

        self.loss_G_total_variance = self.criterionL1(self.calc_gradient(x=self.real_B),self.calc_gradient(self.fake_B))

        self.loss_G = self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance

        self.loss_G.backward()

    def optimize_parameters(self,epoch):
        # print(epoch)
        self.forward(epoch)
        # update D

        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
