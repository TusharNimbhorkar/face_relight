import os.path as osp
import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.utils_SH import get_shading
from models.skeleton512_rgb import HourglassNet as HourglassNet_RGB
from models.skeleton512 import HourglassNet
from PIL import  Image
import torch

dpr_path = '/home/tushar/data2/DPR/train'
lightFolder_dpr = 'test_data/00/'
lightFolder_3dulight = 'test_data/sh_presets/horizontal'
out_dir = '/home/nedko/face_relight/dbs/examples_v0.3_left'

target_sh_id_dpr = 5 #60
target_sh_id_3dulight = 19#89

os.makedirs(out_dir, exist_ok=True)

class Model:
    def __init__(self, checkpoint_path, lab, resolution, dataset_name, sh_const=1.0, name=''):
        self.checkpoint_path = checkpoint_path
        self.lab = lab
        self.resolution = resolution
        self.model = None
        self.sh_const = sh_const
        self.name=name

        if dataset_name == 'dpr':
            self.sh_path = lightFolder_dpr
            self.target_sh = target_sh_id_dpr
        elif dataset_name == '3dulight':
            self.sh_path = lightFolder_3dulight
            self.target_sh = target_sh_id_3dulight

model_lab_pretrained = Model('/home/nedko/face_relight/models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'
model_lab_3dulight_04 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.4/model_256_lab_3dulight_v0.4/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.4')
model_lab_3dulight_03 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.3/model_256_lab_3dulight_v0.3/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.3')
model_lab_3dulight_02 = Model('/home/tushar/data2/checkpoints/model_256_3dudataset_lab/model_256_3dudataset_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.2')
model_rgb_3dulight_02 = Model('/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight', name='RGB 3DULight v0.2')
model_lab_dpr_10k = Model('/home/tushar/data2/checkpoints/model_256_dprdata10k_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', name='LAB DPR 10K')

model_objs = [
    model_lab_pretrained,
    model_lab_3dulight_02,
    model_lab_3dulight_03,
    model_lab_dpr_10k
]

# checkpoint_src = '/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.3/model_256_lab_3dulight_v0.3/14_net_G.pth'
# checkpoint_tgt = '/home/tushar/data2/checkpoints/model_256_3dudataset_lab/model_256_3dudataset_lab/14_net_G.pth' #'/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth' #'/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'


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

def test(my_network, input_img, lab=True, sh_id=0, sh_constant=1.0, res=256, sh_path=lightFolder_3dulight):
    img = input_img
    row, col, _ = img.shape
    # img = cv2.resize(img, size_re)
    img = np.array(Image.fromarray(img).resize((res, res), resample=Image.LANCZOS))
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


    sh = np.loadtxt(osp.join(sh_path, 'rotate_light_{:02d}.txt'.format(sh_id)))
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


for model_obj in model_objs:
    model_obj.model = load_model(model_obj.checkpoint_path, lab=model_obj.lab)

dpr_dirs = sorted(glob.glob(osp.join(dpr_path, '*')))

for dir in dpr_dirs:
    orig_path = osp.join(dir, dir.rsplit('/',1)[-1] + '_05.png')
    # left_path = osp.join(dir, dir.rsplit('/', 1)[-1] + '_00.png')
    orig_img = cv2.imread(orig_path)
    # left_img = cv2.imread(left_path)

    if orig_img.shape[0] > 256:
        orig_img = cv2.resize(orig_img, (256,256))

    results = [orig_img]
    for model_obj in model_objs:
        result_img = test(model_obj.model, orig_img, lab=model_obj.lab, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=model_obj.target_sh, sh_path=model_obj.sh_path)

        if result_img.shape[0]>256:
            result_img = cv2.resize(result_img, (256,256))

        cv2.putText(result_img, model_obj.name, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
        results.append(result_img)

    # tgt_result = cv2.resize(tgt_result, (256,256))

    out_img = np.concatenate(results, axis=1)
    cv2.imwrite(osp.join(out_dir, orig_path.rsplit('/', 1)[-1]), out_img)
    print(dir)
    # plt.imshow(out_img[:,:,::-1])
    # plt.show()