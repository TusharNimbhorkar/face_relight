'''
    this is a simple test file
'''

# trained_model_03.t7
# 14_net_G.pth
# bestcoesf=1.3 and 1.0
import sys

sys.path.append('models')
sys.path.append('utils')
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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

import joblib

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="checkpoint")
# ap.add_argument("-s", "--second", required=True,
#                 help="skip")
args = vars(ap.parse_args())

# load the two input images


checkpoint_dir_cmd = args["first"]
# checkpoint_dir_cmd = 'models/trained/trained_model_03.t7'#args["first"]

skip_c = 0#int(args["second"])
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

def id_to_degrees(id):
    return (int(id) - 7) * 15

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



'''
# original saved file with DataParallel
state_dict = torch.load(checkpoint_dir_cmd)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
my_network.load_state_dict(new_state_dict)
my_network.cuda()
my_network.train(False)

'''
print(checkpoint_dir_cmd)
my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
my_network.cuda()
my_network.train(False)

lightFolder = 'test_data/01/'

sh_vals = ['07']

test_dir = '/home/tushar/data2/data/mpie'


test_dir1 = test_dir


segment = False
# segment[:,:,0]

old_people = os.listdir(test_dir1)


persons = os.listdir(test_dir)


# sh_costant_list = [0.9,1.1,1.2,1.3,1.4,1.5,1.6]
# sh_costant_list = [0.7,0.75,0.8,0.85]
#sh_costant_list = [1.0]
# sh_costant_list = [0.6]



sh_poly_path_temp = 'models/poly_%s_7_new.joblib'
sh_linear_path_temp = 'models/linear_%s_7_new.joblib'


# load the two input images
from_id_list = [8,9,10,6,5,4]
IMS = ['_08.png','_09.png','_10.png','_06.png','_05.png','_04.png',]

device = "cuda"




for ii in range(len(from_id_list)):
    from_id = from_id_list[ii]

    # to_id = from_id_list[ii]

    # from_id = 7
    # to_id = from_id_list[ii]

    to_id = 7

    # print(from_id)
    if from_id == 7:
        pose = id_to_degrees(to_id)
        # print(sh_poly_path)
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
    print('TEST: ', intensity_mul)



    sh_constant_ = intensity_mul

    front_number = IMS[ii]

    # IMS = ['_08.png','_09.png','_10.png','_06.png','_05.png','_04.png',]
    # IMS = ['_10.png']
    # for front_number in IMS:
    overall_error_2 = 0.0
    number = 0.0
    number_seg = 0.0

    overall_error_1 = 0.0

    lowest_error = 9999999
    max_mse = 0

    ##############################################################
    img_dirs = '/home/tushar/data2/face-parsing.PyTorch/res/mpie_segment/'
    people_ = os.listdir(img_dirs)
    peoples = []
    for jj in sorted(people_):
        os.path.join(img_dirs, jj, jj + '_07.png')
        im1_ = cv2.imread(os.path.join(img_dirs, jj, jj + '_07.png'), 0)
        im2_ = cv2.imread(os.path.join(img_dirs, jj, jj + front_number), 0)
        bool_compare = (im1_ == im2_).astype(np.int)
        value = (bool_compare.shape[0] * bool_compare.shape[1]) - np.sum(bool_compare)
        if value > 10000:
            peoples.append(jj)
    ##############################################################


    for per in persons:
        
        person_dir = os.path.join(test_dir, per)
        # from
        # side_im = os.path.join(person_dir, per + '_07.png')
        side_im = os.path.join(person_dir, per + front_number)

        # To
        # front_im = os.path.join(person_dir, per + front_number)
        front_im = os.path.join(person_dir, per + '_07.png')


        exists_ims_side = cv2.imread(side_im) is not None
        exists_ims_front = cv2.imread(front_im) is not None

        if os.path.exists(side_im) and os.path.exists(front_im) and exists_ims_front and exists_ims_side:

            for sh_v in sh_vals:

                saveFolder = os.path.join('runresult', checkpoint_dir_cmd.split('/')[-2], sh_v)

                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder)

                dir_ims = 'test_data/relight_constant'
                ims = os.listdir(dir_ims)
                if sh_v == '07':
                    sh_constant = 0.7

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

                    # outputImg, _, _, _ = my_network(inputL, outputSH*1.3, skip_c)
                    # outputImg, _, _, _ = my_network(inputL, outputSH, skip_c)
                    outputImg, _, _, _ = my_network(inputL, outputSH*sh_constant_, skip_c)

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
                    # #cv2.imwrite(os.path.join(saveFolder, side_im[:-4] + '_{:02d}.jpg'.format(i)), resultLab)
                    # cv2.imwrite(os.path.join('07_10.jpg'.format(i)), resultLab)

                    #######################################################################
                    segment_path_ear = os.path.join('/home/tushar/data2/face-parsing.PyTorch/res/', 'mpie_segment',
                                                    per,
                                                    per + front_number)
                    im1 = cv2.imread(segment_path_ear,0)
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
                        img_side_copy = cv2.resize(img_side_copy,(128,128))
                        segment_im = cv2.resize(segment_im,(128,128))
                        resultLab = cv2.resize(resultLab,(128,128))

                    current_mse_all = mse_all_seg(np.multiply(img_side_copy,segment_im), np.multiply(resultLab,segment_im),segment_im)
                    if current_mse_all>max_mse:
                        max_mse=current_mse_all
                        # print('max  ',per, '  err  ',max_mse)

                    overall_error_1 = overall_error_1 + current_mse_all

                    number_seg=number_seg+1

                    current_mse_all = mse_all(img_side_copy, resultLab)
                    overall_error_2 = overall_error_2 + current_mse_all
                    number = number+1
                    # print(number)



    print(front_number,sh_constant_)
    print("number of files: ",number)
    # print("\nrmse(al): ", (overall_error_2/number))
    print("rmse(al_segment): ", (overall_error_1/number_seg))




