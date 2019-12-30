'''
    this is a simple test file for calculating rmse on segments/full image. Only one predicted-GT pair at a time
'''

# trained_model_03.t7
# 14_net_G.pth
import sys

sys.path.append('models')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np
import argparse
import math

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="checkpoint")
ap.add_argument("-s", "--second", required=True,
                help="skip")
args = vars(ap.parse_args())

# load the two input images


# checkpoint_dir_cmd = args["first"]
# checkpoint_dir_cmd = 'models/trained/trained_model_03.t7'
checkpoint_dir_cmd = 'models/trained/14_net_G_BS8_DPR7.pth'
skip_c = int(args["second"])
# imageB = cv2.imread(args["second"])

'''

begin
'''
def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    rmse = math.sqrt(err)
    return rmse


def mse_all(i1,i2):
    mse_r = mse(i1[:, :, 0], i2[:, :, 0])
    mse_g = mse(i1[:, :, 1], i2[:, :, 1])
    mse_b = mse(i1[:, :, 2], i2[:, :, 2])

    return (mse_r+mse_g+mse_b)/3.0

def mse_seg(imageA, imageB,seg):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # print(float(np.sum(seg[:,:,0])))
    err /= float(np.sum(seg[:,:,0]))
    # err /= float(imageA.shape[0] * imageA.shape[1])
    rmse = math.sqrt(err)
    return rmse


def mse_all_seg(i1,i2,seg):
    mse_r = mse_seg(i1[:, :, 0], i2[:, :, 0],seg)
    mse_g = mse_seg(i1[:, :, 1], i2[:, :, 1],seg)
    mse_b = mse_seg(i1[:, :, 2], i2[:, :, 2],seg)

    return (mse_r+mse_g+mse_b)/3.0



'''
end

'''


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
my_network.train(True)

lightFolder = 'test_data/01/'

sh_vals = ['07']

test_dir = '/home/tushar/face_relight/substet_eval/transfer_q_2x'
# test_dir = '/home/tushar/face_relight/substet_eval/transfer_batch_2x'
# test_dir = '/home/tushar/face_relight/substet_eval/transfer_batch_fullres'


test_dir1 = '/home/tushar/face_relight/substet_eval/transfer_batch_1xhalf'
# test_dir = '/home/tushar/face_relight/substet_eval/transfer_batch_1xhalf'


segment = False
# segment[:,:,0]

old_people = os.listdir(test_dir1)


persons = os.listdir(test_dir)
overall_error_2 = 0.0
number = 0.0
number_seg = 0.0

overall_error_1 = 0.0

from_list = ['04','05','06','07','08','09','10',]
# from_ = '07'
to_ = '07'


for from_ in from_list:

    for per in persons:
        if per in old_people:
            person_dir = os.path.join(test_dir, per)
            # from
            side_im = os.path.join(person_dir, per + '_' + from_ +'.png')
            # To
            front_im = os.path.join(person_dir, per + '_'+to_+'.png')
            if os.path.exists(side_im) and os.path.exists(front_im):

                for sh_v in sh_vals:

                    saveFolder = os.path.join('runresult', checkpoint_dir_cmd.split('/')[-2], sh_v)

                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)

                    dir_ims = 'test_data/relight_constant'
                    ims = os.listdir(dir_ims)
                    if sh_v == '07':
                        sh_constant = 0.7

                    # side_im = '/home/tushar/face_relight/mp_subset/side/epoch014_real_A.png'
                    # front_im = '/home/tushar/face_relight/mp_subset/front/epoch014_real_B.png'

                    im_path = front_im
                    img = cv2.imread(im_path)
                    img_front_copy = cv2.imread(im_path)
                    row, col, _ = img.shape
                    img = cv2.resize(img, (512, 512))
                    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

                    inputL = Lab[:, :, 0]
                    inputL = inputL.astype(np.float32) / 255.0
                    inputL = inputL.transpose((0, 1))
                    inputL = inputL[None, None, ...]
                    inputL = Variable(torch.from_numpy(inputL).cuda())

                    im_path = side_im
                    img1 = cv2.imread(im_path)
                    img_side_copy = cv2.imread(im_path)
                    row1, col1, _ = img1.shape
                    img1 = cv2.resize(img1, (512, 512))
                    Lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)

                    inputL1 = Lab1[:, :, 0]
                    inputL1 = inputL1.astype(np.float32) / 255.0
                    inputL1 = inputL1.transpose((0, 1))
                    inputL1 = inputL1[None, None, ...]
                    inputL1 = Variable(torch.from_numpy(inputL1).cuda())

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

                        sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
                        sh = Variable(torch.from_numpy(sh).cuda())

                        _, _, outputSH, _ = my_network(inputL1, sh, skip_c)

                        # outputImg, _, _, _ = my_network(inputL, outputSH*(10.0/7.0), skip_c)
                        outputImg, _, _, _ = my_network(inputL, outputSH, skip_c)

                        '''sh_viz'''
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
                        # cv2.imwrite(os.path.join('_light_{:02d}_07_front.png'.format(i)), shading)
                        '''end'''

                        outputImg = outputImg[0].cpu().data.numpy()
                        outputImg = outputImg.transpose((1, 2, 0))
                        outputImg = np.squeeze(outputImg)  # *1.45
                        outputImg = (outputImg * 255.0).astype(np.uint8)
                        Lab[:, :, 0] = outputImg
                        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
                        resultLab = cv2.resize(resultLab, (col, row))
                        compare_ = np.hstack((img_side_copy,resultLab,img_front_copy))
                        cv2.imwrite(os.path.join(saveFolder, from_+'_'+to_+ '_{:02d}.jpg'.format(i)), compare_)
                        print(saveFolder, from_+'_'+to_+ '_{:02d}.jpg')
                        # cv2.imwrite(os.path.join('f_s.jpg'.format(i)), resultLab)



                        segment_im = cv2.imread(os.path.join(person_dir, 'mask.png'))
                        segment_im[:, :, 1] = np.copy(segment_im[:, :, 0])
                        segment_im[:, :, 2] = np.copy(segment_im[:, :, 0])
                        segment_im[segment_im > 0] = 1

                        current_mse_all = mse_all_seg(np.multiply(img_side_copy,segment_im), np.multiply(resultLab,segment_im),segment_im)


                        overall_error_1 = overall_error_1 + current_mse_all

                        number_seg=number_seg+1

                        current_mse_all = mse_all(img_side_copy, resultLab)
                        overall_error_2 = overall_error_2 + current_mse_all
                        number = number+1
                        print(number)
                        '''
                        if current_mse_all<35:
                            overall_error_1 = overall_error_1 + current_mse_all
                            number_seg=number_seg+1
    
                        current_mse_all = mse_all(img_side_copy, resultLab)
                        # if current_mse_all<40.:
                        if current_mse_all < 35:
                            overall_error_2 = overall_error_2 + current_mse_all
                            number = number+1
                        print(number)
                        '''


    print("\nnumber of files: ",number)
    print("\nrmse(al): ", (overall_error_2/number))
    print("\nrmse(al_segment): ", (overall_error_1/number_seg))




