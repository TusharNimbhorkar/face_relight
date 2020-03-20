import os.path as osp
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_SH import get_shading
from models.skeleton512_rgb import HourglassNet as HourglassNet_RGB
from models.skeleton512 import HourglassNet
from PIL import  Image
import torch

dpr_path = '/home/tushar/data2/DPR/train'
lightFolder = 'test_data/00/'
sh_constant = 1

checkpoint_src = '/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth'
checkpoint_tgt = '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'

def load_model(checkpoint_dir_cmd, lab=True):
    if lab:
        my_network = HourglassNet()
    else:
        my_network = HourglassNet_RGB()

    print(checkpoint_dir_cmd)
    my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
    my_network.cuda()
    my_network.train(False)
    return my_network

def gen_norm():
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
    return normal, valid

def test(my_network, input_img, lab=True, sh_id=0):
    img = input_img
    row, col, _ = img.shape
    # img = cv2.resize(img, size_re)
    img = np.array(Image.fromarray(img).resize((256, 256), resample=Image.LANCZOS))
    # cv2.imwrite('1.png',img)
    if lab:
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
    else:
        inputL = img
        inputL = inputL.astype(np.float32)
        inputL = inputL / 255.0
        inputL = inputL.transpose((2, 0, 1))
        inputL = inputL[None, ...]

    inputL = torch.autograd.Variable(torch.from_numpy(inputL).cuda())


    sh = np.loadtxt(osp.join(lightFolder, 'rotate_light_{:02d}.txt'.format(sh_id)))
    sh = sh[0:9]
    sh = sh * sh_constant
    # --------------------------------------------------
    # rendering half-sphere
    sh = np.squeeze(sh)
    normal, valid = gen_norm()
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = torch.autograd.Variable(torch.from_numpy(sh).cuda())

    # millis_start = int(round(time.time() * 1000))

    outputImg, _, outputSH, _ = my_network(inputL, sh, 0)

    # outputImg, _, outputSH, _ = my_network(inputL, outputSH, skip_c)

    # millis_after = int(round(time.time() * 1000))
    # elapsed = millis_after - millis_start
    # print('MILISECONDS:  ', elapsed)
    # time_avg += elapsed
    # count = count + 1
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)  # *1.45
    outputImg = (outputImg * 255.0).astype(np.uint8)


    if lab:
        Lab[:, :, 0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    else:
        resultLab = outputImg

    return resultLab


src_model = load_model(checkpoint_src, lab=False)
tgt_model = load_model(checkpoint_tgt, lab=True)

dpr_dirs = sorted(glob.glob(osp.join(dpr_path, '*')))

for dir in dpr_dirs:
    orig_path = osp.join(dir, dir.rsplit('/',1)[-1] + '_05.png')
    left_path = osp.join(dir, dir.rsplit('/', 1)[-1] + '_00.png')
    orig_img = cv2.imread(orig_path)
    left_img = cv2.imread(left_path)

    src_result = test(src_model, orig_img, lab=False)
    tgt_result = test(tgt_model, orig_img, lab=True)

    orig_img = cv2.resize(orig_img, (256,256))
    left_img = cv2.resize(left_img, (256,256))

    out_img = np.concatenate((orig_img, src_result, tgt_result), axis=1)
    print(dir)
    plt.imshow(out_img[:,:,::-1])
    plt.show()