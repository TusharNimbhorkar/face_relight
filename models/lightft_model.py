import torch
import sys
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
sys.path.append('.')
from relight_model512 import *
from relight_model1024 import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
class lightftModel(BaseModel):
    def name(self):
        return 'lightftModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1','G_MSE','G_total_variance', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D', 'G1']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # TODO::  Initialise the network???
        self.netG1 = HourglassNet()
        self.netG1.load_state_dict(torch.load(os.path.join('/home/tushar/Ilumination_gan/models/trained/trained_model_03.t7')))

        # todo check this again
        self.netG1.train(True)

        self.netG = HourglassNet_1024(self.netG1).cuda()
        # todo check this again
        self.netG.train(True)
        # networks.init_weights(self.netG)



        # networks.init_weights(self.netG)
        if self.isTrain:
            # Todo: maybe change this or check it
            self.netD = networks.define_D(2, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.mseloss = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters())
            self.optimizer_D = torch.optim.Adam(self.netD.parameters())

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        #todo: check this

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_AL = input['AL'].to(self.device)
        self.real_BL = input['BL'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        # print(self.real_C.shape)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):

        self.fake_B, self.face_feat_A,self.fake_AL, self.face_feat_B = \
            self.netG(self.real_A, self.real_BL, 0, oriImg=self.real_D)
        # out_img, out_feat, out_light, out_feat_ori


    def calc_gradient(self,x):

        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).cuda())
        G_x = conv1(Variable(x)).data.view(1, 1024, 1024)
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).cuda())
        G_y = conv2(Variable(x)).data.view(1, 1024, 1024)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        return G


    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_C, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_C, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_C, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) #* self.opt.lambda_L1

        self.loss_G_MSE = self.mseloss(self.fake_AL , self.real_AL )

        self.loss_G_total_variance = self.criterionL1(self.calc_gradient(x=self.real_B),self.calc_gradient(self.fake_B))
        print(self.loss_G_total_variance)



        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance

        self.loss_G_feat = self.criterionL1(self.face_feat_A, self.face_feat_B) * 0.5
        self.loss_G = self.loss_G + self.loss_G_feat

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
