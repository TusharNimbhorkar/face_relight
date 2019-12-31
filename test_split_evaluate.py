'''
    this is a simple test file
'''
import sys

sys.path.append('models')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np
import argparse

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="checkpoint")
ap.add_argument("-s", "--second", required=True,
                help="skip")
args = vars(ap.parse_args())


checkpoint_dir_cmd = args["first"]
skip_c = int(args["second"])

checkpoint_dir_our = 'models/trained/14_net_G_BS8_DPR7.pth'

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x ** 2 + z ** 2)
valid = mag <= 1
y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
normal = np.reshape(normal, (-1, 3))
# -----------------------------------------------------------------

modelFolder = 'trained_model/'

# load model
from skeleton512 import *

my_network = HourglassNet()
print(checkpoint_dir_cmd)
my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
my_network.cuda()
my_network.train(False)

our_network = HourglassNet()
print(checkpoint_dir_our)
our_network.load_state_dict(torch.load(checkpoint_dir_our))
our_network.cuda()
our_network.train(False)


lightFolder = 'test_data/01/'
dataroot = '/home/tushar/DPR_data/skel'
sh_vals = ['07']#, '09', '10']
list_im = []
for sh_v in sh_vals:

    saveFolder = os.path.join('runresult_transfer', checkpoint_dir_cmd.split('/')[-2], sh_v)

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    with open('splits/test.lst') as f:
        lines = f.read().splitlines()
    for line in lines:
        list_im.append([os.path.join(dataroot, 'train', line.split(' ')[0], line.split(' ')[1]),
                             os.path.join(dataroot, 'train', line.split(' ')[0], line.split(' ')[2])])


    # dir_ims = 'test_data/2x_MP'
    # ims = os.listdir(dir_ims)
    time_avg = 0
    count = 0.0
    if sh_v == '07':
        sh_constant = 0.7
    if sh_v == '10':
        sh_constant = 1.0
    if sh_v == '09':
        sh_constant = 0.9

    for im in list_im:
        im_path = im
        img = cv2.imread(im_path)
        img_copy = cv2.imread(im_path)
        img_copy = cv2.resize(img_copy, (512, 512))
        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
        inputL = Variable(torch.from_numpy(inputL).cuda())

        for i in range(1):
            sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
            sh = sh[0:9]
            sh = sh * sh_constant
            # --------------------------------------------------
            # rendering half-sphere
            sh = np.squeeze(sh)
            shading = get_shading(normal, sh)
            value = np.percentile(shading, 95)
            ind = shading > value
            shading[ind] = value
            shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
            shading = (shading * 255.0).astype(np.uint8)
            shading = np.reshape(shading, (256, 256))
            shading = shading * valid

            # cv2.imwrite(os.path.join(saveFolder, 'light_{:02d}.png'.format(i)), shading)
            sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
            sh = Variable(torch.from_numpy(sh).cuda())

            millis_start = int(round(time.time() * 1000))

            _, _, outputSH, _ = my_network(inputL, sh, 0)
            outputImg_og, _, _, _ = my_network(inputL, outputSH, 0)

            outputImg, _, outputSH_our, _ = our_network(inputL, outputSH, 0)

            ############################################################################################
            y = torch.Tensor.cpu(outputSH).detach().numpy()
            sh = np.squeeze(y)
            shading = get_shading(normal, sh)
            value = np.percentile(shading, 95)
            ind = shading > value
            shading[ind] = value
            shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
            shading = (shading * 255.0).astype(np.uint8)
            shading = np.reshape(shading, (256, 256))
            shading = shading * valid
            cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_light_{:02d}_og.png'.format(i)), shading)

            y = torch.Tensor.cpu(outputSH_our).detach().numpy()
            sh = np.squeeze(y)
            shading = get_shading(normal, sh)
            value = np.percentile(shading, 95)
            ind = shading > value
            shading[ind] = value
            shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
            shading = (shading * 255.0).astype(np.uint8)
            shading = np.reshape(shading, (256, 256))
            shading = shading * valid
            cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_light_{:02d}_our.png'.format(i)), shading)
            ############################################################################################



            millis_after = int(round(time.time() * 1000))
            elapsed = millis_after - millis_start
            print('MILISECONDS:  ', elapsed)

            outputImg = outputImg[0].cpu().data.numpy()
            outputImg = outputImg.transpose((1, 2, 0))
            outputImg = np.squeeze(outputImg)
            outputImg = (outputImg * 255.0).astype(np.uint8)
            Lab[:, :, 0] = outputImg
            resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
            resultLab = cv2.resize(resultLab, (col, row))
            img_copy = cv2.resize(img_copy, (col, row))

            outputImg_og = outputImg_og[0].cpu().data.numpy()
            outputImg_og = outputImg_og.transpose((1, 2, 0))
            outputImg_og = np.squeeze(outputImg_og)
            outputImg_og = (outputImg_og * 255.0).astype(np.uint8)
            Lab[:, :, 0] = outputImg_og
            resultLab_og = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
            resultLab_og = cv2.resize(resultLab_og, (col, row))

            cv2.imwrite(os.path.join(saveFolder, \
                                     im[:-4] + '_{:02d}.jpg'.format(i)), np.hstack((img_copy,resultLab,resultLab_og)))
