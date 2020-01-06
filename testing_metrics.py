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
import cv2
import math

ap = argparse.ArgumentParser()
ap.add_argument("-fc", "--first_checkpoint", required=True,
                help="checkpoint")

ap.add_argument("-s", "--second", required=True,
                help="skip")

ap.add_argument("-t", "--third", default=False,type=bool,
                help="data parallel or not")
args = vars(ap.parse_args())

checkpoint_dir_cmd = args["first_checkpoint"]

skip_c = int(args["second"])
dataparallel_bool = args["third"]


# checkpoint_dir_our = 'models/trained/14_net_G_BS8_DPR7.pth'


'''
begin MSE FUNCTION
'''
def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    rmse = math.sqrt(err)
    return rmse

def mse_sh(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # err /= float(imageA.shape[0])
    # rmse = math.sqrt(err)
    return err

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
sphere_segment = valid+np.zeros((256,256))
sphere_segment = sphere_segment.astype(np.uint8)

# -----------------------------------------------------------------

# modelFolder = 'trained_model/'

# load model
from skeleton512 import *

if dataparallel_bool:
    my_network = HourglassNet()
    # original saved file with DataParallel
    state_dict = torch.load(checkpoint_dir_cmd)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    my_network.load_state_dict(new_state_dict)
else:
    my_network = HourglassNet()
    print(checkpoint_dir_cmd)
    my_network.load_state_dict(torch.load(checkpoint_dir_cmd))

my_network.cuda()
my_network.train(False)

lightFolder = 'test_data/01/'
dataroot = '/home/tushar/DPR_data/skel'
# dataroot = '/home/tushar/data2/DPR'
sh_vals = ['07']  # , '09', '10']
list_im = []

overall_mse = 0.0
count_number = 0.0


for sh_v in sh_vals:
    saveFolder = os.path.join('runresult_transfer', checkpoint_dir_cmd.split('/')[-2], sh_v)

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    with open('splits/test.lst') as f:
        lines = f.read().splitlines()
    for line in lines:
        list_im.append([os.path.join(dataroot, 'train', line.split(' ')[0], line.split(' ')[1]),
                        os.path.join(dataroot, 'train', line.split(' ')[0], line.split(' ')[2])])

    time_avg = 0
    count = 0.0

    if sh_v == '07':
        sh_constant = 0.7
    if sh_v == '10':
        sh_constant = 1.0
    if sh_v == '09':
        sh_constant = 0.9

    for im in list_im:
        im_path = im[0]
        im_path_target = im[1]

        # *****************source_im*************************
        img = cv2.imread(im_path)
        # img_copy = cv2.imread(im_path)
        # img_copy = cv2.resize(img_copy, (512, 512))
        # print(im_path)
        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
        inputL = Variable(torch.from_numpy(inputL).cuda())

        # *****************target_im***************************

        img_target = cv2.imread(im_path)
        # img_copy_target = cv2.imread(im_path_target)
        # img_copy_target = cv2.resize(img_copy_target, (512, 512))
        row_target, col_target, _ = img_target.shape
        img_target = cv2.resize(img_target, (512, 512))
        Lab_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2LAB)

        inputL_target = Lab_target[:, :, 0]
        inputL_target = inputL_target.astype(np.float32) / 255.0
        inputL_target = inputL_target.transpose((0, 1))
        inputL_target = inputL_target[None, None, ...]
        inputL_target = Variable(torch.from_numpy(inputL_target).cuda())



        del_item_source = im[0].split('_')[-1][:-4]
        del_item_target = im[1].split('_')[-1][:-4]

        SL_path = im[0][:-6] + 'light_' + del_item_source + '.txt'
        TL_path = im[1][:-6] + 'light_' + del_item_target + '.txt'

        # Source_light
        sh = np.loadtxt(SL_path)
        sh = sh[0:9]
        sh = sh * sh_constant
        # --------------------------------------------------
        # rendering half-sphere
        sh_input = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        shading = (shading * 255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading_vis_source_light = shading * valid

        # cv2.imwrite(os.path.join(saveFolder, 'light_{:02d}.png'.format(i)), shading)
        sh_source = np.reshape(sh_input, (1, 9, 1, 1)).astype(np.float32)
        sh_source_tensor = Variable(torch.from_numpy(sh_source).cuda())



        # target
        sh = np.loadtxt(TL_path)
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
        shading_vis_target_light = shading * valid

        # cv2.imwrite(os.path.join(saveFolder, 'light_{:02d}.png'.format(i)), shading)
        sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
        sh_target_tensor = Variable(torch.from_numpy(sh).cuda())


        # forward pass
        # _, _, outputSH, _ = my_network(inputL, sh, 0)


        predicted_img_auth, _, pred_sh_auth, _ = my_network(inputL, sh_target_tensor, 0)

        # predicted_img_our, _, pred_sh_our, _ = my_network(inputL, sh_target_tensor, 0)


        ############################################################################################


        y = torch.Tensor.cpu(pred_sh_auth).detach().numpy()
        sh_pred = np.squeeze(y)
        shading = get_shading(normal, sh_pred)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        shading = (shading * 255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading_pred_auth = shading * valid

        temp_mse_sh = mse_sh(sh_input,sh_pred)

        overall_mse = overall_mse+temp_mse_sh
        count_number = count_number+1



        # cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_light_{:02d}_og.png'.format(i)), shading)
        '''
        y = torch.Tensor.cpu(pred_sh_our).detach().numpy()
        sh = np.squeeze(y)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        shading = (shading * 255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading_pred_ours = shading * valid
        '''
        ###########################################################################################
        # sh_comparison = np.hstack((shading_vis_source_light,shading_pred_auth,shading_pred_ours))
        ###########################################################################################



        # cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_light_{:02d}_our.png'.format(i)), shading)
        # cv2.imwrite( os.path.join(saveFolder, im_path.split(('/'))[-1].split('.')[0] + '_sh_pred.png'), sh_comparison)

        '''

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

        cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_{:02d}.jpg'.format(i)),
                    np.hstack((img_copy, resultLab, resultLab_og)))
        '''


print('MSE_sh',(overall_mse/count_number))