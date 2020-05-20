import argparse
import copy
import os.path as osp
import glob
import cv2
import imutils
import numpy as np
import os
from enum import Enum
import matplotlib.pyplot as plt
from commons.common_tools import FileOutput
from data.base_dataset import get_simple_transform
from models.common.data import img_to_input, output_to_img
from utils.utils_SH import get_shading, SH_basis
from models.skeleton512_rgb import HourglassNet as HourglassNet_RGB
from models.skeleton512 import HourglassNet
# for 1024 skeleton
from models.skeleton1024 import HourglassNet as HourglassNet_512_1024
from models.skeleton1024 import HourglassNet_1024

from PIL import  Image
import torch
import dlib

# for segmentation
from demo_segment.model import BiSeNet
import torchvision.transforms as transforms

class BlendEnum(Enum):
    NONE = "none"
    POISSON = "poisson"
    RENDER_ONLY = "render_only"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default='test_data/test_images', required=False,
	help="Input Directory")
ap.add_argument("-o", "--output", default='outputs/test', required=False,
	help="Output Directory")
ap.add_argument("-d", "--device", default='cuda', required=False, choices=['cuda:0', 'cuda:1', 'cuda','cpu'],
	help="Device")
ap.add_argument("-b", "--blend_mode", default=BlendEnum.NONE, required=False, choices=[blend for blend in BlendEnum],
	help="Blending mode")
ap.add_argument("-f", "--force_images", required=False, action="store_true",
	help="Force generating images")
ap.add_argument("-c", "--crops_only", required=False, action="store_true",
	help="Output cropped faces")
ap.add_argument("-s", "--segment", required=False, action="store_true",
	help="Apply segmentation")
ap.add_argument("-t", "--test", required=False, action="store_true",
	help="Remove text labels and original image comparison")
args = vars(ap.parse_args())

device = args['device']
blend_mode = BlendEnum(args["blend_mode"])
enable_forced_image_out = args['force_images']
enable_segment = args['segment']
enable_face_boxes = args['crops_only']
enable_test_mode = args['test']

lightFolder_dpr = 'test_data/00/'
lightFolder_3dulight = 'test_data/sh_presets/horizontal'
lightFolder_3dulight_shfix = 'test_data/sh_presets/horizontal_shfix2'
lightFolder_dpr_rotated = 'test_data/00_conv'
lightFolder_dpr_test = 'test_data/sh_presets/test_dpr'
lightFolder_3dul_test = 'test_data/sh_presets/test_3dul'

model_dir = osp.abspath('./demo/data/model')
out_dir = args["output"]#'/home/nedko/face_relight/dbs/comparison'

target_sh_id_dpr = list(range(72)) + [71]*20#60#5 #60
target_sh_id_3dulight = list(range(90 - 22 - 45, 90 - 22 + 1))#75 # 19#89

min_video_frames = 10
min_resolution = 1024

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
        paths = sorted(glob.glob(osp.join(self.dir, '*.png')) + glob.glob(osp.join(self.dir, '*.jpg')))
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
    def __init__(self, checkpoint_path, input_mode, resolution, dataset_name, sh_const=1.0, name='', model_1024=False, blend_mode=blend_mode, model_neutral=False, intensity = None):
        self.checkpoint_path = checkpoint_path
        self.input_mode = input_mode
        self.resolution = resolution
        self.model = None
        self.sh_const = sh_const
        self.name=name
        self.device = device ##TODO
        self.model_1024=model_1024
        self.blend_mode = blend_mode
        self.model_neutral = model_neutral
        self.intensity = intensity

        self.transform_src = get_simple_transform(grayscale=False)

        if dataset_name == 'dpr':
            self.sh_path = lightFolder_dpr
            self.target_sh = target_sh_id_dpr
        elif dataset_name == 'dpr_rot':
            self.sh_path = lightFolder_dpr_rotated
            self.target_sh = target_sh_id_dpr
        elif dataset_name == 'dpr_rot_test':
            self.sh_path = lightFolder_dpr_test
            self.target_sh = [0,1,2]
        elif dataset_name == '3dul_test':
            self.sh_path = lightFolder_3dul_test
            self.target_sh = [0,0,0]
        elif dataset_name == '3dulight':
            self.sh_path = lightFolder_3dulight
            self.target_sh = target_sh_id_3dulight
        elif dataset_name == '3dulight_shfix2':
            self.sh_path = lightFolder_3dulight_shfix
            self.target_sh = target_sh_id_3dulight

    def __call__(self, input_img, target_sh, *args, **kwargs):
        target_sh = torch.autograd.Variable(torch.from_numpy(target_sh).to(self.device))

        input_img_tensor = img_to_input(input_img, self.input_mode, transform=self.transform_src)
        # if self.input_mode:
        #     Lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
        #     input_img = Lab[:, :, 0]
        #     input_img = input_img.astype(np.float32) / 255.0
        #     input_img = input_img.transpose((0, 1))
        #     input_img = input_img[None, None, ...]
        # else:
        #     input_img = input_img
        #     input_img = input_img.astype(np.float32)
        #     input_img = input_img / 255.0
        #     input_img = input_img.transpose((2, 0, 1))
        #     input_img = input_img[None, ...]

        input_img_tensor = input_img_tensor[None, ...]

        torch_input_img = torch.autograd.Variable(input_img_tensor.to(self.device))

        model_output = self.model(torch_input_img, target_sh, *args)
        output_img, _, output_sh, _ = model_output

        output_img = output_to_img(output_img, self.input_mode, input_img=input_img)

        return output_img, output_sh

    def instance(self,**kwargs):
        instance = copy.copy(self)
        for (name, val) in kwargs:
            instance.name = val

# dataset_test = DatasetDefault('path/to/files')
dataset_3dulight_v0p8 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.8_256/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_stylegan_v0p2 = Dataset3DULightGT('/home/nedko/face_relight/dbs/stylegan_v0.2_256/train', n_samples=5, n_samples_offset=0)

outputs_path = '/home/nedko/face_relight/outputs/'
outputs_remote_path = '/home/nedko/face_relight/outputs/remote/outputs/'

# model_256_lab_stylegan_0.1_10k_debug
model_lab_stylegan_02_256_10k_intensity_debug_int0 = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_debug/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k Blender intensity=0', intensity=0)
model_lab_stylegan_02_256_10k_intensity_debug_int1 = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_debug/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k Blender intensity=1', intensity=1)
model_lab_stylegan_02_256_10k_intensity_debug_int2 = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_debug/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k Blender intensity=2', intensity=2)

# model_256_labfull_stylegan_0.1_10k_neutral
model_lab_stylegan_01_neutral_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_neutral/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.1 256 Neutral \n10k Blender intensity', model_neutral=True)
model_lab_stylegan_04_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.4_10k/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.4 256 \n10k Blender intensity')
model_lab_stylegan_02_256_10k_intensity_debug = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_debug/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k Blender intensity')
model_lab_stylegan_02_neutral_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.2_10k_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 Neutral \n10k Blender intensity', model_neutral=True)
model_l_stylegan_02_neutral_256_10k_intensity = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k_intensity/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.2 256 Neutral \n10k Blender intensity', model_neutral=True)
model_l_stylegan_03_neutral_256_10k = Model(outputs_path + 'model_256_lab_stylegan_0.3_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.3 256 Neutral 10k', model_neutral=True)
model_l_stylegan_02_neutral_256_10k = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.2 256 Neutral 10k', model_neutral=True)
model_lab_stylegan_01_256_10k = Model(outputs_remote_path + 'model_256_lab_3dulab_0.8_10k_l+ab/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.1 256 10k')
model_l_stylegan_01_256_10k_ep1fix = Model(outputs_path + 'model_256_lab_sgan_0.1_10k_ep1fix/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 10k ep1fix')

model_l_stylegan_01_neutral_256_10k = Model(outputs_remote_path + 'model_256_lab_neutral_sgan_0.1_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 Neutral 10k', model_neutral=True)
model_l_stylegan_01_256_10k = Model(outputs_path + 'model_256_lab_sgan_0.1_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 10k')
model_l_3dulight_08_256_10k_shfix2_test = Model(outputs_path + 'model_256_lab_3dulab_0.8_test_10k_2/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 10k')
# model_l_3dulight_08_256_10k_shfix2_test = Model(outputs_path + 'model_256_lab_3dulab_0.8_test_10k/13_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 10k New')
model_l_3dulight_08_256_full_shfix2 = Model(outputs_path + 'model_256_lab_3dulab_0.8_test/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 30k')
model_l_stylegan_01_256_neutral_full = Model(outputs_path + 'model_neutral_256_lab_stylegan_v0.1/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 Neutral 256 30k', model_neutral=True)
model_l_stylegan_01_256_full = Model(outputs_path + 'model_256_lab_stylegan_v0.1/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 30k')
model_l_3dulight_08_256_full_bs7 = Model(outputs_path + 'model_256_lab_3dulight_v0.8_full_bs7/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 256 30k BS7')
model_l_3dulight_08_256_10k_shfix2 = Model(outputs_path + 'model_256_lab_3dulight_v0.8_shfix2/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL v0.8 256 10k SHFIX2')
model_l_3dulight_08_512_30k_bs7 = Model(outputs_remote_path + 'model_512_lab_3dulight_v0.8_full_bs7_bs7/14_net_G.pth', input_mode='L', resolution=512, dataset_name='3dulight', name='L 3DUL v0.8 512 30k BS7')
model_l_3dulight_08_1024_10k = Model('/home/tushar/data2/face_relight/outputs/model_1024_3du_v08_lab_10k_lg59/13_net_G.pth', input_mode='L', resolution=1024, dataset_name='3dulight', name='L DPR v0.8 1024 10k', model_1024=True)
model_l_dpr_08_1024_30k = Model('/home/tushar/data2/checkpoints_debug/model_fulltrain_dpr7_gan_BS7_1024/10_net_G.pth', input_mode='L', resolution=1024, dataset_name='dpr', sh_const = 0.7,name='L DPR v0.8 1024 30k')
model_l_3dulight_08_512_30k_render = Model(outputs_remote_path + 'model_512_lab_3dulight_v0.8_full_bs7/14_net_G.pth', input_mode='L', resolution=512, dataset_name='3dulight', name='L 3DUL v0.8 512 30k Render', blend_mode=BlendEnum.RENDER_ONLY)
model_l_3dulight_08_512_30k_noblend = Model(outputs_remote_path + 'model_512_lab_3dulight_v0.8_full_bs7/14_net_G.pth', input_mode='L', resolution=512, dataset_name='3dulight', name='L 3DUL v0.8 512 30k No Blend', blend_mode=BlendEnum.NONE)
model_l_3dulight_08_512_30k = Model(outputs_remote_path + 'model_512_lab_3dulight_v0.8_full_bs7/14_net_G.pth', input_mode='L', resolution=512, dataset_name='3dulight', name='L 3DUL v0.8 512 30k')
model_l_3dulight_08_1024_10k_third = Model('/home/tushar/data2/face_relight/outputs/model_1024_3du_v08_lab_third/14_net_G.pth', input_mode='L', resolution=1024, dataset_name='3dulight', name='L 3DUL v0.8 1024 10k Old', model_1024=True)
model_l_3dulight_08_bs20 = Model(outputs_remote_path + 'model_256_lab_3dulight_v0.8_full_bs7/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 30k bs20')
model_l_3dulight_08_seg_face = Model(outputs_remote_path + '3dulight_v0.8_256_seg_face/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 10k Segment')
model_l_dpr_seg = Model('/home/tushar/data2/checkpoints_debug/model_fulltrain_dpr7_mse_sumBS20_ogsegment/14_net_G.pth', input_mode='L', resolution=256, dataset_name='dpr', sh_const = 0.7, name='L DPR v0.8 10k Segment')
model_l_3dulight_08_seg = Model('/home/tushar/data2/face_relight/outputs/model_256_3du_v08_lab_seg/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 10k Segment')
model_l_3dulight_08_full_seg = Model(outputs_remote_path + '3dulight_v0.8_256_full/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 30k Segment +hair')
model_l_3dulight_08_bs16 = Model(outputs_remote_path + '3dulight_v0.8_256_bs16/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 bs16')
model_rgb_3dulight_08_full = Model(outputs_path + 'model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v0.8_full/14_net_G.pth', input_mode='RGB', resolution=256, dataset_name='3dulight', name='RGB 3DUL v0.8 30k')
model_rgb_3dulight_08 = Model(outputs_path + 'model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', input_mode='RGB', resolution=256, dataset_name='3dulight', name='RGB 3DUL v0.8')
model_l_3dulight_08_full = Model(outputs_path + 'model_256_lab_3dulight_v0.8_full/model_256_lab_3dulight_v0.8_full/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8 30k')
model_rgb_3dulight_08_rot_test = Model(outputs_path + 'model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', input_mode='RGB', resolution=256, dataset_name='dpr_rot_test', name='RGB 3DUL v0.8')
model_rgb_3dulight_08_3dul_test = Model(outputs_path + 'model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', input_mode='RGB', resolution=256, dataset_name='3dul_test', name='RGB 3DUL v0.8')
model_l_3dulight_08 = Model(outputs_path + 'model_256_lab_3dulight_v0.8/model_256_lab_3dulight_v0.8/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DUL v0.8')
model_l_3dulight_05_shfix = Model(outputs_path + 'model_256_lab_3dulight_v0.5_shfix/model_256_lab_3dulight_v0.5_shfix/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight', name='L 3DULight v0.5 SHFIX')
model_l_dpr_10k = Model('/home/tushar/data2/checkpoints/model_256_dprdata10k_lab/14_net_G.pth', input_mode='L', resolution=256, dataset_name='dpr', sh_const = 0.7, name='L DPR 10K')
model_l_dpr_512_30k = Model('/home/tushar/data2/checkpoints_debug/model_fulltrain_dpr7_mse_sumBS20/14_net_G.pth', input_mode='L', resolution=512, dataset_name='dpr', sh_const = 0.7, name='L DPR 512 30K')
model_l_pretrained = Model('models/trained/trained_model_03.t7', input_mode='L', resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'

model_objs = [
    model_l_stylegan_02_neutral_256_10k,
    model_l_stylegan_02_neutral_256_10k_intensity,
    model_lab_stylegan_02_neutral_256_10k_intensity,
    model_lab_stylegan_01_neutral_256_10k_intensity,
    # model_lab_stylegan_01_256_10k,
    # model_lab_stylegan_02_256_10k_intensity_debug,
    # model_lab_stylegan_04_256_10k_intensity,
    # model_lab_stylegan_02_256_10k_intensity_debug_int0,
    # model_lab_stylegan_02_256_10k_intensity_debug_int1,
    # model_lab_stylegan_02_256_10k_intensity_debug_int2
]

# dataset = dataset_stylegan_v0p2
dataset = DatasetDefault(args["input"])

min_resolution = np.min([min_resolution] + [model_obj.resolution for model_obj in model_objs])

# checkpoint_src = '/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.3/model_256_lab_3dulight_v0.3/14_net_G.pth'
# checkpoint_tgt = '/home/tushar/data2/checkpoints/model_256_3dudataset_lab/model_256_3dudataset_lab/14_net_G.pth' #'/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth' #'/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'


detector = dlib.get_frontal_face_detector()
lmarks_model_path = osp.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(lmarks_model_path)

if enable_segment:
    n_classes = 19
    segment_model = BiSeNet(n_classes=n_classes)
    segment_model.to(device)
    segment_model_path = osp.join(model_dir, 'face_parsing.pth')
    segment_model.load_state_dict(torch.load(segment_model_path))
    segment_model.eval()

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

R_90 = R(np.deg2rad(90))


def Rx(x, sx):
    return np.array([
            [sx, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])

def Ry(y, sy):
    return np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, sy, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])

def Rz(z, sz):
    return np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, sz]
        ])

def R(x, y, z, sx=1, sy=1, sz=1):
    return Rz(z, sz) @ Ry(y, sy) @ Rx(x, sx)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

segment_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

def segment(img_, device):
    with torch.no_grad():
        h, w, _ = img_.shape
        image = cv2.resize(img_, (512, 512), interpolation=cv2.INTER_AREA)
        img = segment_norm(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = segment_model(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        output_img = vis_parsing_maps(image, parsing, stride=1,h=h,w=w)
    return output_img


def resize_pil(image, width=None, height=None, inter=Image.LANCZOS):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = np.array(
        Image.fromarray(image.astype(np.uint8)).resize(dim, resample=Image.LANCZOS))
    # return the resized image
    return resized

def preprocess(img, device, enable_segment):
    img = np.array(img)
    orig_size = img.shape

    if np.max(img.shape[:2]) > 1024:
        if img.shape[0] < img.shape[1]:
            img_res = resize_pil(img, width=1024)
        else:
            img_res = resize_pil(img, height=1024)
    else:
        img_res = img

    resize_ratio = orig_size[0] / img_res.shape[0]
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    rects, scores, idx = detector.run(gray, 1, -1)

    loc = [0, 0]

    if len(rects) > 0:

        rect_id = np.argmax(scores)
        rect = rects[rect_id]
        # rect = rects[0]
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # (x, y, w, h) = rect_to_bb(rect)
        e0 = np.array(shape[38])
        e1 = np.array(shape[43])
        m0 = np.array(shape[48])
        m1 = np.array(shape[54])

        x_p = e1 - e0
        y_p = 0.5 * (e0 + e1) - 0.5 * (m0 + m1)
        c = 0.5 * (e0 + e1) - 0.1 * y_p
        s = np.max([4.0 * np.linalg.norm(x_p), 3.6 * np.linalg.norm(y_p)])
        xv = x_p - np.dot(R_90, y_p)
        xv /= np.linalg.norm(xv)
        yv = np.dot(R_90, y_p)

        s *= resize_ratio
        c[0] *= resize_ratio
        c[1] *= resize_ratio

        c1_ms = np.max([0, int(c[1] - s / 2)])
        c1_ps = np.min([img.shape[0], int(c[1] + s / 2)])
        c0_ms = np.max([0, int(c[0] - s / 2)])
        c0_ps = np.min([img.shape[1], int(c[0] + s / 2)])

        top = -np.min([0, int(c[1] - s / 2)])
        bottom = -np.min([0, img.shape[0] - int(c[1] + s / 2)])
        left = -np.min([0, int(c[0] - s / 2)])
        right = -np.min([0, img.shape[1] - int(c[0] + s / 2)])

        loc[0] = int(c1_ms)
        loc[1] = int(c0_ms)

        img = img[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

        if enable_segment:
            mask = segment(img, device)
        else:
            mask = None
        # mask = mask[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if enable_segment:
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        border = [top, bottom, left, right]

        crop_sz = img.shape
        if np.max(img.shape[:2]) > 1024:
            img = cv2.resize(img, (1024, 1024))

            if enable_segment:
                mask = cv2.resize(mask, (1024, 1024))
    else:
        img = None
        mask = None
        crop_sz = None
        border = None

    return img, mask, loc, crop_sz, border


def handle_output(out_img, col, row, mask, img_p, img_orig, loc, crop_sz, border, enable_face_boxes, item_name, sh_id, sh, blend_mode):
    render_data_dir = '/home/nedko/face_relight/dbs/rendering'
    model_data_dir = '/home/nedko/face_relight/outputs/test_bg'
    out_dir = '/home/nedko/face_relight/outputs/test_bg_replace'
    masks_fname = 'mask_full.png'
    norms_fname = 'normals_warped.png'

    mask_path = osp.join(render_data_dir, item_name, masks_fname)
    norms_path = osp.join(render_data_dir, item_name, norms_fname)

    if sh_id is not None:
        render_path = osp.join(render_data_dir, item_name, '%04d.jpg' % (sh_id * 2))
    else:
        render_path = 'empty'
    print(mask_path, norms_path, render_path)

    if osp.exists(mask_path) and osp.exists(norms_path) and osp.exists(render_path) and blend_mode in [BlendEnum.POISSON, BlendEnum.RENDER_ONLY]:
        mask_rendering = (np.asarray(cv2.imread(mask_path)) / 255)
        # mask = mask_rendering[loc[0]:loc[0] + mask.shape[0], loc[1]:loc[1] + mask.shape[1]]
        norms = np.asarray(cv2.imread(norms_path)).astype(np.float)
        norms[:, :, 0] = norms[:, :, 0]  / 127.5 - 1
        norms[:, :, 1] = norms[:, :, 1]  / 127.5 - 1
        norms[:, :, 2] = (norms[:, :, 2] - 127.5) / 127.5
        # norms = norms[:,:,::-1]
        norms[:,:,1], norms[:,:,2] = np.copy(norms[:,:,2]), np.copy(norms[:,:,1])

        r = R(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), sx=-1)[np.newaxis,...]

        shp = norms.shape
        norms= np.reshape(norms, (norms.shape[0]*norms.shape[1],3, 1))

        norms = r@norms
        norms = np.reshape(norms, (shp[0], shp[1], 3))
        img_rendered = cv2.imread(render_path)
    else:
        mask_rendering = None
        img_rendered = None
        norms = None

    # mask = (mask/255.0)[:,:, np.newaxis]
    result = cv2.resize(out_img, (col, row))

    # do something here
    # make a gauss blur

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TODO add original image which will be background  CHECK AGAIN  !!!!!!!!!!!

    if mask is not None and img_rendered is None:
        background = np.copy(img_p)
        foreground = np.copy(result)
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(mask, foreground)

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1 - mask, background)

        # Add the masked foreground and background.
        out_img = cv2.add(foreground, background)
    else:
        out_img = np.copy(result)

    out_img = cv2.resize(out_img, (crop_sz[1], crop_sz[0]))



    if not enable_face_boxes:
        top, bottom, left, right = border

        right = -right
        bottom = -bottom

        if bottom == 0:
            bottom = None

        if right == 0:
            right = None

        out_img = out_img[top:bottom, left:right]

        if img_rendered is not None:
            img_overlayed = np.copy(img_orig)
            img_overlayed[loc[0]:loc[0] + out_img.shape[0], loc[1]:loc[1] + out_img.shape[1]] = out_img
            mask_rendering_crop = mask_rendering[loc[0]:loc[0] + out_img.shape[0], loc[1]:loc[1] + out_img.shape[1]]

            # basis = SH_basis([[1,1,1]])
            # print(norms.shape)
            sh = np.reshape(sh, (9,1))
            sh[0,0]*=3
            shading = get_shading(np.reshape(norms, (norms.shape[0]*norms.shape[1], 3)), sh)
            # value = np.percentile(shading, 95)
            # ind = shading > value
            # shading[ind] = value

            shading = np.reshape(shading, (norms.shape[0], norms.shape[1], 1))
            shaded = np.array(img_orig).astype(np.float) * shading
            img_rendered = (255*(shaded/np.max(shaded))).astype(np.uint8)

            # shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
            # shading = np.reshape(shading, (norms.shape[0], norms.shape[1],1))
            # img = (np.array(img_orig).astype(np.float)*shading).astype(np.uint8)

            # mask_rendering = cv2.GaussianBlur(mask_rendering, (15, 15), 15, 15)

            # background = cv2.multiply(1-mask_rendering, np.array(img_rendered).astype(np.float))
            # foreground = cv2.multiply(mask_rendering, np.array(img_overlayed).astype(np.float))
            # img_overlayed = cv2.add(background, foreground)


            out_img = out_img.astype(np.uint8)
            img_rendered = img_rendered.astype(np.uint8)

            # Create an all white mask
            mask = 255 * np.ones(out_img.shape, out_img.dtype)

            # The location of the center of the src in the dst
            width, height, channels = out_img.shape
            # loc[0]: loc[0] + out_img.shape[0], loc[1]: loc[1] + out_img.shape[1]
            center = (loc[1] + int(out_img.shape[1]/2), loc[0] + int(out_img.shape[0]/2))
            # center = ( int(out_img.shape[1] / 2), int(out_img.shape[0] / 2))


            if blend_mode == BlendEnum.POISSON:
                from commons.torch_tools import Chronometer
                chron = Chronometer(torch.device('cpu'))
                chron.tick()
                # out_img_expanded = np.copy(img_orig)
                # out_img_expanded[loc[0]:loc[0] + out_img.shape[0], loc[1]:loc[1] + out_img.shape[1]] = out_img
                # mask_rendering_extended = np.zeros(out_img_expanded.shape)
                # mask_rendering_extended[loc[0]:loc[0] + out_img.shape[0], loc[1]:loc[1] + out_img.shape[1]] = mask_rendering_crop
                # mask_rendering_crop = mask_rendering_extended

                mask_rendering_crop_gt0 = mask_rendering_crop > 0
                mask_rendering_crop[mask_rendering_crop_gt0]=255

                # mask_rendering_crop = np.abs(mask_rendering_crop - 255)
                # mask_rendering_crop_gt0 = ~mask_rendering_crop_gt0

                mask_rendering_crop = mask_rendering_crop.astype(np.uint8)
                occurences = np.where(mask_rendering_crop_gt0)
                mask_loc = [np.min(occurences[0]), np.min(occurences[1])]
                mask_sz = [np.max(occurences[0])-mask_loc[0]+1, np.max(occurences[1])-mask_loc[1]+1]
                center = (loc[1] + mask_loc[1] + int(mask_sz[1]/2), loc[0] + mask_loc[0] + int(mask_sz[0]/2))
                print(center, mask_sz, mask_rendering_crop.shape, mask_rendering_crop_gt0.shape)

                img_overlayed = cv2.seamlessClone(out_img, img_rendered, mask_rendering_crop, center, cv2.NORMAL_CLONE)
                print('TIME', chron.tock())
            elif blend_mode == BlendEnum.RENDER_ONLY:
                img_overlayed = img_rendered
        else:
            img_overlayed = np.copy(img_orig)
            img_overlayed[loc[0]:loc[0] + out_img.shape[0], loc[1]:loc[1] + out_img.shape[1]] = out_img
    else:
        img_overlayed = out_img



    # # print(loc[0] - 10,loc[0] + outImage.shape[0] + 10, loc[1] - 10,loc[1] + outImage.shape[1] + 10)
    #
    #
    # # blending fr the bounding box
    # img1 = np.ones_like(img_overlayed)
    #
    # img1[loc[0]+2:loc[0] + out_img.shape[0]-2, loc[1]+2:loc[1] + out_img.shape[1]-2] = 0
    # mask = cv2.bitwise_not(img1)
    # mask[mask < 255] = 0
    #
    #
    # # blending
    #
    # mask = cv2.GaussianBlur(mask, (7, 7), 7, 7)
    # # Normalize the alpha mask to kee   p intensity between 0 and 1
    # mask = mask.astype(float) / 255
    # background = np.copy(img_orig)
    # foreground = np.copy(img_overlayed)
    # foreground = foreground.astype(float)
    # background = background.astype(float)
    #
    # # Multiply the foreground with the alpha matte
    # foreground = cv2.multiply(mask, foreground)
    #
    # # Multiply the background with ( 1 - alpha )
    # background = cv2.multiply(1 - mask, background)
    #
    # # Add the masked foreground and background.
    # out_img = cv2.add(foreground, background)
    #
    # #
    # # cv2.imwrite(filepath, outImage)

    return img_overlayed


def vis_parsing_maps(im, parsing_anno, stride, h=None, w=None):
    im = np.array(im)
    alpha_2 = np.zeros((h, w, 3))
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # MASK
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (w, h))
    vis_parsing_anno[vis_parsing_anno == 16] = 0
    # vis_parsing_anno[vis_parsing_anno==17]=0
    vis_parsing_anno[vis_parsing_anno == 14] = 0
    vis_parsing_anno[vis_parsing_anno > 0] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closing = cv2.morphologyEx(vis_parsing_anno, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    new_img = np.zeros_like(closing)  # step 1
    for val in np.unique(closing)[1:]:  # step 2
        mask = np.uint8(closing == val)  # step 3
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
        new_img[labels == largest_label] = val

    vis_parsing_anno = new_img.copy()

    # alpha_2 = cv2.imread(segment_path_ear)
    alpha_2[:, :, 0] = np.copy(vis_parsing_anno)
    alpha_2[:, :, 1] = np.copy(vis_parsing_anno)
    alpha_2[:, :, 2] = np.copy(vis_parsing_anno)
    kernel = np.ones((10, 10), np.uint8)
    alpha_2 = cv2.erode(alpha_2, kernel, iterations=1)
    alpha_2 = cv2.GaussianBlur(alpha_2, (29, 29), 15, 15)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha_2 = alpha_2.astype(float) / 255
    return alpha_2

def load_model(checkpoint_dir_cmd, device, input_mode='L', model_1024=False, model_neutral=False):
    if input_mode in ['L', 'LAB']:
        nc_img = 3 if input_mode == 'LAB' else 1
        if model_1024:
            my_network_512 = HourglassNet_512_1024(16)
            my_network = HourglassNet_1024(my_network_512, 16)
        else:
            my_network = HourglassNet(enable_target=not model_neutral, ncImg=nc_img)
    else:
        my_network = HourglassNet_RGB()

    print(checkpoint_dir_cmd)
    my_network.load_state_dict(torch.load(checkpoint_dir_cmd))
    my_network.to(device)
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

def test(my_network, input_img, sh_id=0, sh_constant=1.0, res=256, sh_path=lightFolder_3dulight, sh_fname=None, extra_ops={}, intensity=None):
    img = input_img
    row, col, _ = img.shape
    # img = cv2.resize(img, size_re)
    img = np.array(Image.fromarray(img).resize((res, res), resample=Image.LANCZOS))
    # cv2.imwrite('1.png',img)


    if sh_fname is None:
        sh_fname = 'rotate_light_{:02d}.txt'.format(sh_id)

    sh = np.loadtxt(osp.join(sh_path, sh_fname))
    sh = sh[0:9]
    if intensity is not None:
        sh[0] = intensity
    sh = sh * sh_constant
    # --------------------------------------------------
    # rendering half-sphere
    sh = np.squeeze(sh)
    # normal, valid = gen_norm()
    # shading = get_shading(normal, sh)
    # value = np.percentile(shading, 95)
    # ind = shading > value
    # shading[ind] = value
    # shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    # shading = (shading * 255.0).astype(np.uint8)
    # shading = np.reshape(shading, (256, 256))
    # shading = shading * valid

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)

    output_img, output_sh = my_network(img, sh, 0, **extra_ops)

    return output_img, sh

def draw_label(img, label):
    img_width = img.shape[1]
    canvas = (np.ones((50, img_width, 3)) * 255).astype(np.uint8)
    canvas[:, -1:, :] = 0
    canvas[:, :1, :] = 0
    for i, row in enumerate(label.split('\n')):
        cv2.putText(canvas, row, (5, 20*(i+1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
    img = np.concatenate((img, canvas), axis=0)

    return img

enable_target_sh = False
for model_obj in model_objs:
    if not model_obj.model_neutral:
        enable_target_sh = True
    model_obj.model = load_model(model_obj.checkpoint_path, model_obj.device, input_mode=model_obj.input_mode, model_1024=model_obj.model_1024, model_neutral=model_obj.model_neutral)

for orig_path, out_fname, gt_data in dataset.iterate():

    sh_path_dataset = None
    gt_path = None
    max_res_deviation = 300

    if gt_data is not None:
        gt_path, sh_path_dataset = gt_data

    orig_img = cv2.imread(orig_path)

    orig_img_proc, mask, loc, crop_sz, border = preprocess(orig_img, device, enable_segment)
    is_face_found = orig_img is not None

    # orig_img_proc = resize_pil(orig_img_proc, height=min_resolution)

    if not is_face_found:
        print('No face found: ', orig_path)
        continue

    if enable_test_mode:
        results = []
    else:
        if enable_face_boxes:
            orig_img_show = orig_img_proc
        else:
            orig_img_show = orig_img

        if orig_img_show.shape[0] > min_resolution:
            orig_img_show = resize_pil(orig_img_show, height=min_resolution)

        orig_img_show = draw_label(orig_img_show, 'Original')

        results = [orig_img_show]

    if gt_path is not None:
        gt_img = cv2.imread(gt_path)

        # if gt_img.shape[0] > min_resolution:
        gt_img = resize_pil(gt_img, height=orig_img_proc.shape[0])

        print(gt_img.shape, min_resolution)
        if not enable_test_mode:
            gt_img = draw_label(gt_img, 'Ground Truth')
        results.append(gt_img)

    if sh_path_dataset is None and enable_target_sh:
        min_sh_list_len = min([len(model_obj.target_sh) for model_obj in model_objs])
    else:
        min_sh_list_len = 1

    enable_video_out = min_sh_list_len > min_video_frames and not enable_forced_image_out

    if enable_video_out:
        video_out = FileOutput(osp.join(out_dir, out_fname.rsplit('.',1)[0]+'.avi'))

    enable_res_deviation_warning = False
    for model_obj in model_objs:
        if max_res_deviation is not None and (model_obj.resolution - orig_img_proc.shape[0]>max_res_deviation or model_obj.resolution - orig_img_proc.shape[0]>max_res_deviation):
            enable_res_deviation_warning = True
            break

    if enable_res_deviation_warning:
        continue

    for sh_idx in range(min_sh_list_len):
        results_frame = []
        for model_obj in model_objs:

            if sh_path_dataset is None:
                sh_path = model_obj.sh_path
                target_sh = model_obj.target_sh[sh_idx]
                sh_fname = None
            else:
                sh_path, sh_fname = sh_path_dataset.rsplit('/', 1)
                target_sh = None

            extra_ops={}

            result_img, sh = test(model_obj, orig_img_proc, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=target_sh, sh_path=sh_path, sh_fname=sh_fname, extra_ops=extra_ops, intensity=model_obj.intensity)

            result_img = handle_output(result_img, orig_img_proc.shape[1], orig_img_proc.shape[0], mask, orig_img_proc, orig_img, loc, crop_sz, border, enable_face_boxes, orig_path.rsplit('/', 1)[-1].rsplit('.', 1)[0], target_sh, sh, model_obj.blend_mode)

            if result_img.shape[0]>min_resolution:
                result_img = resize_pil(result_img, height=min_resolution)

            result_img = np.ascontiguousarray(result_img, dtype=np.uint8)

            if not enable_test_mode:
                result_img = draw_label(result_img, model_obj.name)
            results_frame.append(result_img)

        # tgt_result = cv2.resize(tgt_result, (256,256))
        print([entry.shape for entry in results+results_frame])
        out_img = np.concatenate(results + results_frame, axis=1)
        print(orig_path, gt_path)

        if enable_video_out:
            video_out.post(out_img)
        else:
            cv2.imwrite(osp.join(out_dir, out_fname.rsplit('.', 1)[0] + '_%03d' % sh_idx + '.' + out_fname.rsplit('.', 1)[1]), out_img)

    if enable_video_out:
        video_out.close()
    # plt.imshow(out_img[:,:,::-1])
    # plt.show()