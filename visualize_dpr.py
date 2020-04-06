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

lightFolder_dpr = 'test_data/00/'
lightFolder_3dulight_shfix = 'test_data/sh_presets/horizontal'
lightFolder_3dulight = 'test_data/sh_presets/horizontal_old'
out_dir = '/home/nedko/face_relight/dbs/comparison_right'

target_sh_id_dpr = 60#5 #60
target_sh_id_3dulight = 70 # 19#89
target_sh_id_3dulight_shfix = 75 # 19#89

min_resolution = 256

os.makedirs(out_dir, exist_ok=True)

class Dataset:
    def __init__(self, dir):
        self.dir = dir

    def iterate(self):
        pass

class DatasetDefault(Dataset):
    ''' Used for generic test sets'''
    def __init__(self, dir):
        super().__init__(dir)

    def iterate(self):
        paths = sorted(glob.glob(osp.join(self.path, '*.png')) + glob.glob(osp.join(self.path, '*.jpg')))
        for path in paths:
            out_fname = path.rsplit('/',1)[-1]
            yield path, out_fname, None

class DatasetDPR(Dataset):
    '''DPR dataset'''
    def __init__(self, dir):
        super().__init__(dir)


    def iterate(self):
        data_dirs = sorted(glob.glob(osp.join(self.dir, '*')))
        for dir in data_dirs:
            orig_path = osp.join(dir, dir.rsplit('/',1)[-1] + '_05.png')
            out_fname = orig_path.rsplit('/',1)[-1]
            yield orig_path, out_fname, None

class Dataset3DULight(Dataset):
    '''3DULight dataset'''
    def __init__(self, dir):
        super().__init__(dir)

    def iterate(self):
        dpr_dirs = sorted(glob.glob(osp.join(self.dir, '*')))
        for dir in dpr_dirs:
            orig_path = osp.join(dir, 'orig.png')
            _, dirname, fname = orig_path.rsplit('/', 2)
            out_fname = dirname + '_' + fname
            yield orig_path, out_fname, None

class Dataset3DULightGT(Dataset3DULight):
    ''' A dataset with ground truth SH'''
    def __init__(self, dir, n_samples=None, n_samples_offset=0):
        super().__init__(dir)
        self.n_samples = n_samples
        if n_samples_offset is not None:
            self.n_samples = n_samples + n_samples_offset

        self.n_samples_offset = n_samples_offset

    def iterate(self):
        dirs = sorted(glob.glob(osp.join(self.dir, '*')))
        for dir in dirs:
            orig_path = osp.join(dir, 'orig.png')
            paths = sorted(glob.glob(osp.join(dir, '*.png')) + glob.glob(osp.join(dir, '*.jpg')))[self.n_samples_offset:self.n_samples]
            for path in paths:
                parent_dir, dirname, fname = path.rsplit('/', 2)
                subname, _ = fname.rsplit('.',1)

                sh_path = osp.join(parent_dir, dirname, 'light_%s_sh.txt' % subname)
                if not osp.exists(sh_path):
                    continue

                out_fname = dirname + '_' + fname
                yield orig_path, out_fname, [path, sh_path]


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
        elif dataset_name == '3dulight_shfix':
            self.sh_path = lightFolder_3dulight_shfix
            self.target_sh = target_sh_id_3dulight_shfix

dataset_3dulight_v0p8 =  Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.8_256/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p7_randfix =  Dataset3DULightGT('/home/tushar/data2/face_relight/dbs/3dulight_v0.7_256_fix/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p6 =  Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.6_256/train', n_samples=5, n_samples_offset=5) # DatasetDPR('/home/tushar/data2/DPR/train')

model_lab_pretrained = Model('/home/nedko/face_relight/models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'
model_rgb_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v0.8_full/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight_shfix', name='RGB 3DULight v0.8 30k')
model_rgb_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight_shfix', name='RGB 3DULight v0.8')
model_lab_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8_full/model_256_lab_3dulight_v0.8_full/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 30k')
# model_lab_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8/model_256_lab_3dulight_v0.8/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8')
model_lab_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8/model_256_lab_3dulight_v0.8/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8')
model_lab_3dulight_07_randfix = Model('/home/tushar/data2/face_relight/outputs/model_256_lab_3dulight_v0.7_random_ns5/model_256_lab_3dulight_v0.7_random_ns5/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.7 RANDFIX')
model_lab_3dulight_07 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.7_dlfix_ns15/model_256_lab_3dulight_v0.7_dlfix_ns15/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.7')
model_lab_3dulight_06 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.6_dlfix_ns15/model_256_lab_3dulight_v0.6_dlfix_ns15/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.6')
model_lab_3dulight_05_shfix = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.5_shfix/model_256_lab_3dulight_v0.5_shfix/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.5 SHFIX')
model_lab_3dulight_05 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.5/model_256_lab_3dulight_v0.5/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.5')
model_lab_3dulight_04 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.4/model_256_lab_3dulight_v0.4/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.4')
model_lab_3dulight_03 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.3/model_256_lab_3dulight_v0.3/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.3')
model_lab_3dulight_02 = Model('/home/tushar/data2/checkpoints/model_256_3dudataset_lab/model_256_3dudataset_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.2')
model_rgb_3dulight_02 = Model('/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight', name='RGB 3DULight v0.2')
model_lab_dpr_10k = Model('/home/tushar/data2/checkpoints/model_256_dprdata10k_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', name='LAB DPR 10K')

model_objs = [
    # model_lab_pretrained,
    model_rgb_3dulight_08,
    model_rgb_3dulight_08_full,
    model_lab_3dulight_08_full
]

dataset = dataset_3dulight_v0p8

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

def test(my_network, input_img, lab=True, sh_id=0, sh_constant=1.0, res=256, sh_path=lightFolder_3dulight, sh_fname=None):
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

    if sh_fname is None:
        sh_fname = 'rotate_light_{:02d}.txt'.format(sh_id)

    sh = np.loadtxt(osp.join(sh_path, sh_fname))
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

for orig_path, out_fname, gt_data in dataset.iterate():

    sh_path_dataset = None
    gt_path = None

    if gt_data is not None:
        gt_path, sh_path_dataset = gt_data

    # orig_path = osp.join(dir, dir.rsplit('/',1)[-1] + '_05.png')
    # left_path = osp.join(dir, dir.rsplit('/', 1)[-1] + '_00.png')
    orig_img = cv2.imread(orig_path)
    # left_img = cv2.imread(left_path)

    if orig_img.shape[0] > min_resolution:
        orig_img = cv2.resize(orig_img, (min_resolution,min_resolution))

    results = [orig_img]

    if gt_path is not None:
        gt_img = cv2.imread(gt_path)

        if gt_img.shape[0] > min_resolution:
            gt_img = cv2.resize(gt_img, (min_resolution, min_resolution))

        cv2.putText(gt_img, 'Ground Truth', (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
        results.append(gt_img)

    for model_obj in model_objs:
        if sh_path_dataset is None:
            sh_path = model_obj.sh_path
            target_sh = model_obj.target_sh
            sh_fname = None
        else:
            sh_path, sh_fname = sh_path_dataset.rsplit('/', 1)
            target_sh = -1

        result_img = test(model_obj.model, orig_img, lab=model_obj.lab, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=target_sh, sh_path=sh_path, sh_fname=sh_fname)

        if result_img.shape[0]>min_resolution:
            result_img = cv2.resize(result_img, (min_resolution,min_resolution))

        result_img = np.ascontiguousarray(result_img, dtype=np.uint8)
        cv2.putText(result_img, model_obj.name, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
        results.append(result_img)

    # tgt_result = cv2.resize(tgt_result, (256,256))

    out_img = np.concatenate(results, axis=1)
    cv2.imwrite(osp.join(out_dir, out_fname), out_img)
    print(orig_path, gt_path)
    # plt.imshow(out_img[:,:,::-1])
    # plt.show()