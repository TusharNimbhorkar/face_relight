# MSE-SUM on SH is mean not sum
# Default model
import torch
import sys

from commons.common_tools import overrides
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

sys.path.append('.')
# from relight_model import *
from .skeleton512 import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os.path as osp

class lightgrad59stackModel(BaseModel):
    def name(self):
        return 'lightgrad59stackModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--enable_neutral', action='store_true', help='Enable or disable input target sh')
            parser.add_argument('--force_load', action='store_true', help='Force loading of weights which does not match this model by removing any tensors that do not match.')

        return parser

    def load_networks(self, epoch):
        """Load models from the disk"""
        for name in self.model_names:
            print(name)
            if name=='G' or name=='D':
                if isinstance(name, str):
                    load_filename = '%s_net_%s.pth' % (epoch, name)

                    if self.load_dir is not None:
                        load_dir = self.load_dir
                    else:
                        load_dir = self.save_dir

                    load_path = osp.join(load_dir, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    loaded_state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(loaded_state_dict, '_metadata'):
                        del loaded_state_dict._metadata


                    if name == 'G':
                        current_state_dict = self.netG.module.state_dict()
                        # print('TEST', current_state_dict['pre_conv.weight'].shape)
                        # print('TEST', current_state_dict['output.weight'].shape)
                    elif name == 'D':
                        current_state_dict = self.netD.module.state_dict()
                        # print('TEST', current_state_dict['model.0.weight'].shape)
                    #
                    if not self.opt.force_load:

                        if name == 'G' and current_state_dict['light.predict_FC1.weight'].shape[1] != loaded_state_dict['light.predict_FC1.weight'].shape[1]:
                            del loaded_state_dict['light.predict_FC1.weight']
                            try:
                                del loaded_state_dict['light.predict_FC1.bias']
                            except:
                                do_nothing=0

                        if name == 'G' and current_state_dict['pre_conv.weight'].shape[1] != loaded_state_dict['pre_conv.weight'].shape[1]:
                            del loaded_state_dict['pre_conv.weight']
                            del loaded_state_dict['pre_conv.bias']
                        #
                        if name == 'G' and current_state_dict['output.weight'].shape[0] != loaded_state_dict['output.weight'].shape[0]:
                            del loaded_state_dict['output.weight']
                            del loaded_state_dict['output.bias']
                        #
                        if name == 'D' and current_state_dict['model.0.weight'].shape[1] != loaded_state_dict['model.0.weight'].shape[1]:
                            del loaded_state_dict['model.0.weight']
                            del loaded_state_dict['model.0.bias']

                    else:
                        keys_to_del = []
                        for key in loaded_state_dict:
                            print(loaded_state_dict[key].shape)
                            if current_state_dict[key].shape != loaded_state_dict[key].shape:
                                keys_to_del.append(key)

                        for key in keys_to_del:
                            del loaded_state_dict[key]


                    for key in list(loaded_state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self._patch_instance_norm_state_dict(loaded_state_dict, net, key.split('.'))

                    net.load_state_dict(loaded_state_dict, strict=False)

                    # sd = net.cpu().state_dict()


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'G_MSE', 'G_total_variance', 'G_feat', 'L1_add', 'G', 'D_real', 'D_fake', 'D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1', 'D2']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.enable_target = not opt.enable_neutral
        self.input_mode = opt.input_mode

        self.nc_light_extra = 0
        if self.input_mode in ['RGB', 'LAB']:
            self.nc_img = 3
        elif self.input_mode in ['L']:
            self.nc_img = 1

        self._set_model_parameters()
        '''
        self.netG = HourglassNet(enable_target=False, ncImg=self.nc_img, ncLightExtra=self.nc_light_extra).to(self.device)
        '''
        # Neutral
        self.netG1 = HourglassNet(enable_target=self.enable_target, ncImg=self.nc_img, ncLightExtra=self.nc_light_extra).to(self.device)

        # Relight
        self.netG2 = HourglassNet(enable_target=True, ncImg=self.nc_img, ncLightExtra=self.nc_light_extra).to(self.device)

        if 'cpu' not in str(self.device):
            self.netG1 = torch.nn.DataParallel(self.netG1, self.opt.gpu_ids)
            self.netG2 = torch.nn.DataParallel(self.netG2, self.opt.gpu_ids)
        self.netG1.train(True)
        self.netG2.train(True)

        if self.isTrain:
            self.epochs_G_only = 0
            '''
            self.netD = networks.define_D(2 * self.nc_img, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            '''
            # neutral
            self.netD1 = networks.define_D(2 * self.nc_img, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # relight
            self.netD2 = networks.define_D(2 * self.nc_img, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(gan_mode='lsgan').to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.mseloss = torch.nn.MSELoss(reduction='mean').to(self.device)

            # initialize optimizers
            self.optimizers = []

            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters())
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters())
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters())
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters())

            self.G_opt = torch.optim.Adam(list(self.netG1.parameters()) + list(self.netG2.parameters()))
            self.D_opt = torch.optim.Adam(list(self.netD1.parameters()) + list(self.netD2.parameters()))

            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)

    def _set_model_parameters(self):
        pass

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_AL = input['AL'].to(self.device)
        self.real_BL = input['BL'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, epoch):

        count_skip = 4
        if epoch > 5:
            count_skip = 9 % epoch
        if epoch >= 10:
            count_skip = 0
        '''
        if epoch <= 10:
            self.fake_B, _, self.fake_AL, _ = self.netG(self.real_A, self.real_BL, count_skip, oriImg=None)

        if epoch > 10:
            self.fake_B, self.face_feat_A, self.fake_AL, self.face_feat_B = self.netG(self.real_A, self.real_BL,
                                                                                      count_skip, oriImg=self.real_D)'''
        # figure out the what needs to be inputed or not
        count_skip = 0
        self.fake_B1, _, self.fake_AL, _ = self.netG1(self.real_A, self.real_BL, count_skip, oriImg=None)
        self.fake_B2, _, self.fake_AL, _ = self.netG2(self.fake_B1, self.real_BL, count_skip, oriImg=None)

        # print(self.fake_B1,self.fake_B2)

    def calc_gradient(self, x):
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        if x.shape[1] == 3:
            if self.input_mode == 'LAB':
                grad_input = x[:,0:1,:, :]
            elif self.input_mode == 'RGB':

                grad_input = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(self.device)

                # 0.2989 * R + 0.5870 * G + 0.1140 * B
                #           B           G           R
                grad_input[:, 0] = x[:, 0]*0.1140 + x[:, 1]*0.5870 + x[:, 2]*0.2989
        else:
            grad_input = x

        conv1 = nn.Conv2d(self.nc_img, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(self.device))
        G_x = conv1(Variable(grad_input))
        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(self.nc_img, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(self.device))
        G_y = conv2(Variable(grad_input))
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

        return G

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B

        # TODO: Check for the inputs to the discriminator.
        ### D1 ###
        fake_AB = torch.cat((self.real_C, self.fake_B1), 1)
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_C, self.real_B), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D1 = (self.loss_D_fake + self.loss_D_real) * 0.5

        ### D2 ###

        fake_AB = torch.cat((self.real_C, self.fake_B2), 1)
        pred_fake = self.netD2(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_C, self.real_B), 1)
        pred_real = self.netD2(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D = self.loss_D1 + self.loss_D2

        self.loss_D.backward()

    def backward_G(self, epoch):
        # First, G(A) should fake the discriminator

        ### G1 loss ###

        fake_AB = torch.cat((self.real_C, self.fake_B1), 1)

        self.loss_G_GAN = 0

        if self.epochs_G_only==0 or epoch>1:
            pred_fake = self.netD1(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.5

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B1)  # * self.opt.lambda_L1

        self.loss_G_MSE = self.mseloss(self.real_AL, self.fake_AL)

        self.loss_G_total_variance = self.criterionL1(self.calc_gradient(x=self.real_B),
                                                      self.calc_gradient(self.fake_B1))

        self.loss_L1_add = self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance

        self.loss_G1 = self.loss_G_GAN + self.loss_L1_add #self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance
        #
        '''#off for a moment
        if epoch > 10:
            self.loss_G_feat = self.mseloss(self.face_feat_A1, self.face_feat_B1) * 0.5
            self.loss_G = self.loss_G + self.loss_G_feat
        else:
            self.loss_G_feat = 0.0'''


        ### G2 Loss ###
        fake_AB = torch.cat((self.real_C, self.fake_B2), 1)

        self.loss_G_GAN = 0

        if self.epochs_G_only == 0 or epoch > 1:
            pred_fake = self.netD2(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.5

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B2)  # * self.opt.lambda_L1

        self.loss_G_MSE = self.mseloss(self.real_AL, self.fake_AL)

        self.loss_G_total_variance = self.criterionL1(self.calc_gradient(x=self.real_B),
                                                      self.calc_gradient(self.fake_B2))

        self.loss_L1_add = self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance

        self.loss_G2 = self.loss_G_GAN + self.loss_L1_add  # self.loss_G_L1 + self.loss_G_MSE + self.loss_G_total_variance
        '''#off for a moment
        if epoch > 10:
            self.loss_G_feat = self.mseloss(self.face_feat_A, self.face_feat_B) * 0.5
            self.loss_G = self.loss_G + self.loss_G_feat
        else:
            self.loss_G_feat = 0.0'''

        self.loss_G = self.loss_G1 + self.loss_G2
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        if self.epochs_G_only!=0 and epoch/self.epochs_G_only<1:
            epoch = 1
        else:
            epoch = epoch - self.epochs_G_only

        self.forward(epoch)

        if self.epochs_G_only==0 or epoch>1:
            # update D
            self.set_requires_grad(self.netD1, True)
            self.set_requires_grad(self.netD2, True)
            # self.optimizer_D.zero_grad()
            self.D_opt.zero_grad()
            self.backward_D()
            self.D_opt.step()

        # update G
        self.set_requires_grad(self.netD1, False)
        self.set_requires_grad(self.netD2, False)
        # self.optimizer_G.zero_grad()
        self.G_opt.zero_grad()
        self.backward_G(epoch)
        self.G_opt.step()
