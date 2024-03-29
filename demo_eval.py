'''
    this is a simple test file to calculate RMSE of MULTIPIE only. 
'''

import sys

sys.path.append('models')

# other modules
import os
import numpy as np
import argparse
import math
from torch.autograd import Variable
import torch
import cv2
import joblib

ap = argparse.ArgumentParser()
ap.add_argument('--frontal', dest='frontal', action='store_true')
ap.set_defaults(feature=True)
args = vars(ap.parse_args())

checkpoint_dir_cmd = 'models/trained/14_net_G_dpr7_mseBS20.pth'

'''
functions to calculate mean squared error
'''


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    rmse = math.sqrt(err)
    return rmse


def mse_seg(imageA, imageB, seg):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(np.sum(seg[:, :, 0]))
    rmse = math.sqrt(err)
    return rmse


def mse_all_seg(i1, i2, seg):
    mse_r = mse_seg(i1[:, :, 0], i2[:, :, 0], seg)
    mse_g = mse_seg(i1[:, :, 1], i2[:, :, 1], seg)
    mse_b = mse_seg(i1[:, :, 2], i2[:, :, 2], seg)

    return (mse_r + mse_g + mse_b) / 3.0


def id_to_degrees(id):
    return (int(id) - 7) * 15


'''
end

'''

modelFolder = 'trained_model/'

# load model
from skeleton512 import *

my_network = HourglassNet(enable_target=True, ncImg=1, ncLightExtra=0, ncLight=27)

current_state_dict = [module.state_dict().keys() for module in my_network.modules()]
loaded_params = torch.load(checkpoint_dir_cmd)
my_network.load_state_dict(loaded_params)
my_network.cuda()
my_network.train(False)


lightFolder = 'test_data/01/'
test_dir = '/home/tushar/data2/folders_not_in_use/data/mpie'
persons = os.listdir(test_dir)
sh_poly_path_temp = 'models/poly_%s_7.joblib'
sh_linear_path_temp = 'models/linear_%s_7.joblib'

# load the two input images
from_id_list = [8, 9, 10, 6, 5, 4]
IMS = ['_08.png', '_09.png', '_10.png', '_06.png', '_05.png', '_04.png', ]
device = "cuda"

for ii in range(len(from_id_list)):

    if args['frontal']:
        from_id = 7
        to_id = from_id_list[ii]
    else:
        from_id = from_id_list[ii]
        to_id = 7

    if from_id == 7:
        pose = id_to_degrees(to_id)
        sh_poly_path = sh_poly_path_temp % 'from'
        sh_linear_path = sh_linear_path_temp % 'from'
    elif to_id == 7:
        pose = id_to_degrees(from_id)
        sh_poly_path = sh_poly_path_temp % 'to'
        sh_linear_path = sh_linear_path_temp % 'to'
    else:
        raise ValueError()

    poly = joblib.load(sh_poly_path)
    lin = joblib.load(sh_linear_path)
    feat = poly.transform([[pose]])
    intensity_mul = lin.predict(feat)[0][0]
    sh_constant_ = intensity_mul

    front_number = IMS[ii]
    number_files = 0.0
    overall_error = 0.0

    for per in persons:

        person_dir = os.path.join(test_dir, per)
        # from
        if args['frontal']:
            side_im = os.path.join(person_dir, per + '_07.png')
            front_im = os.path.join(person_dir, per + front_number)
            from_sh = 0
            to_sh = pose
            # print('pose',pose)

        else:
            side_im = os.path.join(person_dir, per + front_number)
            front_im = os.path.join(person_dir, per + '_07.png')
            from_sh = pose
            to_sh = 0

        exists_ims_side = cv2.imread(side_im) is not None
        exists_ims_front = cv2.imread(front_im) is not None

        if os.path.exists(side_im) and os.path.exists(front_im) and exists_ims_front and exists_ims_side:

            dir_ims = 'test_data/relight_constant'
            ims = os.listdir(dir_ims)

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
                sh = sh
                sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
                sh = Variable(torch.from_numpy(sh).cuda())
                # --------------------------------------------------
                _, _, outputSH, _ = my_network(inputL1, sh, 0)
                outputImg, _, _, _ = my_network(inputL, outputSH * sh_constant_, 0)
                # --------------------------------------------------

                outputImg = outputImg[0].cpu().data.numpy()
                outputImg = outputImg.transpose((1, 2, 0))
                outputImg = np.squeeze(outputImg)  # *1.45
                outputImg = (outputImg * 255.0).astype(np.uint8)
                Lab[:, :, 0] = outputImg
                resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
                resultLab = cv2.resize(resultLab, (col, row))

                #######################################################################
                segment_path_ear = os.path.join('/home/tushar/data2/face-parsing.PyTorch/res/', 'mpie_segment',
                                                per,
                                                per + front_number)
                im1 = cv2.imread(segment_path_ear, 0)
                im1[im1 == 8] = 255
                im1[im1 == 7] = 255
                im1[im1 < 255] = 0
                im1[im1 == 0] = 254
                im1[im1 == 255] = 1
                im1[im1 == 254] = 0
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im1)
                area = []

                for j in range(len(stats)):
                    area.append(stats[j][4])
                sor_area = sorted(area)

                area_1 = area.index(sor_area[-2])
                area_2 = area.index(sor_area[-3])

                labels[labels == area_1] = 255
                labels[labels == area_2] = 255
                labels[labels < 255] = 0
                labels[labels == 0] = 1
                labels[labels == 255] = 0

                ####################################################################################

                segment_im = cv2.imread(os.path.join(person_dir, 'mask.png'))
                first_channel = np.copy(segment_im[:, :, 0])
                first_channel[first_channel < 255] = 0
                first_channel = np.multiply(first_channel, labels)

                segment_im[:, :, 0] = np.copy(first_channel)
                segment_im[:, :, 1] = np.copy(first_channel)
                segment_im[:, :, 2] = np.copy(first_channel)
                segment_im[segment_im > 0] = 1
                resize = True
                if resize:
                    img_side_copy = cv2.resize(img_side_copy, (128, 128))
                    segment_im = cv2.resize(segment_im, (128, 128))
                    resultLab = cv2.resize(resultLab, (128, 128))

                current_mse = mse_all_seg(np.multiply(img_side_copy, segment_im), np.multiply(resultLab, segment_im),
                                          segment_im)
                overall_error = overall_error + current_mse
                number_files = number_files + 1

    print('\nRMSE Score for transfering light of ', from_sh, ' to ', to_sh)
    print("rmse(al_segment): ", (overall_error / number_files))
