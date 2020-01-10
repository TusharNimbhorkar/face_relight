'''
    this is a simple test file
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

from torch.autograd import Variable
from commons.common_tools import Logger, BColors
from torchvision.utils import make_grid
import torch
import time
import cv2
from skeleton512 import *

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights_dir", required=True,
	help="checkpoint")
ap.add_argument("-s", "--skip", required=True,
	help="skip")
args = vars(ap.parse_args())

# load the two input images

log = Logger("Evaluation", tag_color=BColors.Red)

checkpoint_dir_cmd = args["weights_dir"]
skip_c = int(args["skip"])
#imageB = cv2.imread(args["second"])



# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------

# load model
my_network = HourglassNet()
print(checkpoint_dir_cmd)
my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
my_network.cuda()
my_network.train(False)

lightFolder = 'test_data/00/'

sh_vals = ['07']#,'09','10']

for sh_v in sh_vals:

	saveFolder = os.path.join('runresult_test',checkpoint_dir_cmd.split('/')[-2],sh_v)

	if not os.path.exists(saveFolder):
		os.makedirs(saveFolder)

	dir_ims = 'test_data/relight_constant'
	ims = os.listdir(dir_ims)
	time_avg = 0
	count = 0.0
	if sh_v=='07':
		sh_constant=0.7
	if sh_v=='10':
		sh_constant=1.0
	if sh_v=='09':
		sh_constant=0.9

	for im in ims:
		im_path = os.path.join(dir_ims,im)
		img = cv2.imread(im_path)
		row, col, _ = img.shape
		img = cv2.resize(img, (512, 512))
		Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		inputL = Lab[:,:,0]
		inputL = inputL.astype(np.float32)/255.0
		inputL = inputL.transpose((0,1))
		inputL = inputL[None,None,...]
		inputL = Variable(torch.from_numpy(inputL).cuda())

		real_img = "/home/nedko/face_relight/outputs/portrait/aa.jpg"
		real_img = cv2.imread(real_img)
		real_img = cv2.resize(real_img, (512, 512))
		LabR = cv2.cvtColor(real_img, cv2.COLOR_BGR2LAB)

		inputR = LabR[:, :, 0]
		inputR = inputR.astype(np.float32) / 255.0
		inputR = inputR.transpose((0, 1))
		inputR = inputR[None, None, ...]
		inputR = Variable(torch.from_numpy(inputR).cuda())

		avg_losses = {}
		for i in range(36):
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

			millis_start = int(round(time.time() * 1000))

			model=my_network

			outputImg, feat_A, outputSH, feat_B = my_network(inputL, sh, skip_c, inputR)

			criterionL1 = torch.nn.L1Loss().cuda()
			loss_G_feat = criterionL1(feat_A, feat_B) * 0.5
			log.i("Feature loss:",float(loss_G_feat), np.linalg.norm(feat_A.cpu().detach().numpy()))
			# model.forward(15)
			# model.backward_G(15, train_mode=False)
			# losses = model.get_current_losses()
			# for name in losses.keys():
			# 	print(name, losses[name])
			# 	if name not in avg_losses.keys():
			# 		avg_losses[name] = 0
			# 	avg_losses[name] += losses[name]


			# outputImg, _, outputSH, _ = my_network(inputL, outputSH, skip_c)

			millis_after = int(round(time.time() * 1000))
			elapsed = millis_after - millis_start
			print('MILISECONDS:  ', elapsed)
			time_avg += elapsed
			count = count + 1
			outputImg = outputImg[0].cpu().data.numpy()
			outputImg = outputImg.transpose((1, 2, 0))
			outputImg = np.squeeze(outputImg) #*1.45
			outputImg = (outputImg * 255.0).astype(np.uint8)
			Lab[:, :, 0] = outputImg
			resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
			resultLab = cv2.resize(resultLab, (col, row))
			cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_{:02d}.jpg'.format(i)), resultLab)
			# cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_{:02d}.jpg'.format(i)), outputImg)
