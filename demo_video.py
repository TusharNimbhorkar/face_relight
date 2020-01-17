'''
    this is a simple test file to generate two types of video. One is over the surface of spehere and other one is horizoatal light change
'''

import sys

sys.path.append('models')
import os
import numpy as np
import argparse
from torch.autograd import Variable
import torch
import cv2
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True,
                help="input image to relight, --input_image /path/to/image/file")
ap.add_argument("-t", "--type", required=True,
                help="type of change, --type over or --type horizontal")
ap.add_argument("-v", "--video_name", required=True,
                help="name of the output video, --video_name name_of_the_output_video")
ap.add_argument("-fr", "--frame_rate", type=int, default=10,
                help="Frame rate for output video (Default=10), --frame_rate fps")
args = vars(ap.parse_args())
checkpoint_dir_cmd = 'models/trained/14_net_G_dpr7_mseBS20.pth'

# load model
from skeleton512 import *

my_network = HourglassNet()
print(checkpoint_dir_cmd)
my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
my_network.cuda()
my_network.train(False)

if args['type'] == 'horizontal':
    path_light = '00'
if args['type'] == 'over':
    path_light = '01'
lightFolder = os.path.join('test_data/', path_light)

saveFolder = os.path.join('tmp/', )
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
dir_ims = 'test_data/relight_constant'
ims = os.listdir(dir_ims)
im_path = args['input_image']
img = cv2.imread(im_path)
row, col, _ = img.shape
img = cv2.resize(img, (512, 512))
Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

inputL = Lab[:, :, 0]
inputL = inputL.astype(np.float32) / 255.0
inputL = inputL.transpose((0, 1))
inputL = inputL[None, None, ...]
inputL = Variable(torch.from_numpy(inputL).cuda())

for i in range(int(len(os.listdir(lightFolder)) / 2)):
    sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
    sh = sh[0:9]
    sh = sh * 0.8
    sh = np.squeeze(sh)
    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, _, outputSH, _ = my_network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))
    prefix = im_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(saveFolder, prefix + '_{:02d}.jpg'.format(i)), resultLab)

print('GENERATING VIDEO...')
obj_path = 'tmp/'
imlist = sorted(os.listdir(obj_path))
temp_im = cv2.imread(os.path.join(obj_path, imlist[0]))

save_video_folder = 'video_output/'
if not os.path.exists(save_video_folder):
    os.makedirs(save_video_folder)
video_path = os.path.join(save_video_folder, args['video_name'] + args['type'] + '.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args['frame_rate'],
                      (temp_im.shape[1], temp_im.shape[0]))
print('Saving video to: ', video_path)
for j in imlist:
    og_im = os.path.join(obj_path, j)
    og_img = cv2.imread(og_im)
    out.write(og_img)
out.release()
shutil.rmtree('tmp/')
