import argparse
import copy
import os.path as osp
import glob
import cv2
import imutils
import numpy as np
import os
from enum import Enum
from commons.common_tools import def_log as log
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

from utils.utils_data import InputProcessor, resize_pil


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
ap.add_argument("-b", "--blend_mode", default=BlendEnum.NONE.value, required=False, choices=[blend.value for blend in BlendEnum],
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
    def __init__(self, checkpoint_path, input_mode, resolution, dataset_name, sh_const=1.0, name='', model_1024=False,
                 blend_mode=blend_mode, enable_neutral=False, intensity=None, ambience=None, nc_sh=3, enable_sun_diam=False, enable_sun_color=False, enable_amb_color=False, enable_face_tone=False):
        self.checkpoint_path = checkpoint_path
        self.input_mode = input_mode
        self.resolution = resolution
        self.model = None
        self.sh_const = sh_const
        self.name=name
        self.device = device ##TODO
        self.model_1024=model_1024
        self.blend_mode = blend_mode
        self.model_neutral = enable_neutral
        self.intensity = intensity
        self.ambience = ambience
        self.nc_sh = nc_sh
        self.sun_diam = enable_sun_diam
        self.enable_sun_color = enable_sun_color
        self.enable_amb_color = enable_amb_color
        self.enable_face_tone = enable_face_tone

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
        if self.ambience is not None:
            ambience_arr = np.array([self.ambience]).reshape((1,1,1,1))
            target_sh = np.append(target_sh, ambience_arr, axis=1)

        target_sh = target_sh.astype(np.float32)
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
        for (name, val) in kwargs.items():
            instance.__dict__[name] = val

        # print(instance.name, self.name)
        return instance

# dataset_test = DatasetDefault('path/to/files')
dataset_3dulight_v0p8 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.8_256/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_stylegan_v0p2 = Dataset3DULightGT('/home/nedko/face_relight/dbs/stylegan_v0.2_256/train', n_samples=5, n_samples_offset=0)

outputs_path = '/home/nedko/face_relight/outputs/'
outputs_remote_path = '/home/nedko/face_relight/outputs/remote/outputs/'

#model_256_lab_stylegan_0.6.0_10k_neutral_sundiam_new_pcrop
# model_256_lab_stylegan_0.5.2_10k_neutral_ambcolor_allprops_pcrop

# model_256_lab_stylegan_0.3.1_10k_neutral_pcrop_50k
#model_256_lab_stylegan_0.5.2_20k_neutral_pcrop_r30k+r30k
#model_256_lab_stylegan_0.5.2_20k_neutral_pcrop
outputs_path_tushar_ds3 = "/home/tushar/data1/project/face_relight/outputs/"


# variable g set, D set experiment.
model_256_lab_stylegan_031_30k_neutral_pcrop_r30k_r30k = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_30k_neutral_pcrop_r30k+r30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_30k_031_256 \n D: 30k+30k', intensity=0, enable_neutral=True)
model_256_lab_stylegan_031_30k_neutral_pcrop_r50k_r50k = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_30k_neutral_amb_pcrop_r50k+50k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_30k_031_256 \n D: 50k+50k', intensity=0, enable_neutral=True)
model_256_lab_stylegan_031_30k_neutral_pcrop_r70k_r70k = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_30k_neutral_amb_pcrop_r70k+70k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_30k_031_256 \n D: 70k+70k', intensity=0, enable_neutral=True)

model_256_lab_stylegan_031_50k_neutral_pcrop_r30k_r30k = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_50k_neutral_amb_pcrop_r30k+30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_50k_031_256 \n D: 30k+30k', intensity=0, enable_neutral=True)
model_256_lab_stylegan_031_50k_neutral_pcrop_r50k_r50k = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_50k_neutral_amb_pcrop_r50k+50k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_50k_031_256 \n D: 50k+50k', intensity=0, enable_neutral=True)
model_256_lab_stylegan_031_50k_neutral_pcrop_r70k_r70k = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_50k_neutral_amb_pcrop_r70k+70k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='lab_sgan_50k_031_256 \n D: 70k+70k', intensity=0, enable_neutral=True)

model_256_lab_stylegan_031_50k_neutral_pcrop_r30k_r30k_rerun = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_50k_neutral_amb_pcrop_r30k+30k_try2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='RERUN lab_sgan_50k_031_256 \n D: 30k+30k', intensity=0, enable_neutral=True)
model_256_lab_stylegan_031_50k_neutral_pcrop_r50k_r50k_rerun = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_50k_neutral_amb_pcrop_r50k+50k_try2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='RERUN lab_sgan_50k_031_256 \n D: 50k+50k', intensity=0, enable_neutral=True)


## ft model
model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k_ft = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_20k_neutral_pcrop_r30k+r30k_ft/24_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='(FT)L+AB 20k sGAN v0.5.2 256\n D:30k ffhq, 30k SGAN', intensity=0, enable_neutral=True, enable_amb_color=True)

# model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k
## overfit
model_256_lab_stylegan_031_100_neutral_pcrop_r30k_r30k_overfit = Model(outputs_path_tushar_ds3 + 'model_256_lab_stylegan_0.3.1_100_neutral_amb_pcrop_r30k+30k_overfit_try2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='OVERFIT lab_sgan_100_031_256 \n D: 30k+30k', intensity=0, enable_neutral=True)

# rgb model
model_256_rgb_stylegan_031_50k_neutral_pcrop_r30k_r0k = Model(outputs_path_tushar_ds3 + 'model_256_rgb_stylegan_0.3.1_50k_neutral_amb_pcrop_r30k+30k/14_net_G.pth', input_mode='RGB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='RGB _sgan_50k_031_256 \n D: 30k+30k', intensity=0, enable_neutral=True)


model_512_lab_stylegan_052_20k_r30k_30k_neutral_pcrop = Model(outputs_path_tushar_ds3 + 'model_512_lab_stylegan_0.5.2_20k_r30k+30k_neutral_noprops_pcrop/14_net_G.pth', input_mode='LAB', resolution=512, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.5.2 512\n D:30k ffhq, 30k SGAN', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_20k_neutral_pcrop_r30k+r30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.5.2 256\n int=0, Neut, rmix30k, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_052_256_20k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_20k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.5.2 256\n int=0, Neut, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_031_256_70k_neut_pcrop_rmix70k = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_70k_neutral_amb_pcrop_r70k+70k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 70k sGAN v0.5.2 256\n int=0, Neut, rmix70k, pcrop', intensity=0, enable_neutral=True)
# model_lab_stylegan_052_256_70k_neut_pcrop_rmix70k = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_70k_neutral_pcrop_r70k+r70k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 70k sGAN v0.5.2 256\n int=0, Neut, rmix70k, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)
# model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_20k_neutral_pcrop_r30k+r30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.5.2 256\n int=0, Neut, rmix30k, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_052_256_30k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_30k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 30k sGAN v0.5.2 256\n int=0, Neut, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_052_256_30k_neut_pcrop_rmix30k = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_30k_r30k+30k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 30k sGAN v0.5.2 256\n int=0, Neut, rmix30k, pcrop', intensity=0, enable_neutral=True, enable_amb_color=True)

model_lab_stylegan_031_256_30k_neut_pcrop_rmix30k = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_30k_neutral_pcrop_r30k+r30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 30k sGAN v0.3.1 256\n int=0, Neut,rmix30k pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_20k_neut_pcrop_rmix30k = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_20k_neutral_pcrop_r30k+r30k/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.3.1 256\n int=0, Neut,rmix30k pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_20k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_20k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 20k sGAN v0.3.1 256\n int=0, Neut, pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_70k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_70k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 70k sGAN v0.3.1 256\n int=0, Neut, pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_50k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_50k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 50k sGAN v0.3.1 256\n int=0, Neut, pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_081_256_10k_neut_ambcolor_suncolor_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.8.1_30k_neutral_pcrop_nosundiam/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.8.1 256\n10k int=0, Neut ambcol+suncol', intensity=0, enable_neutral=True, enable_amb_color=True, enable_sun_color=True)
model_lab_stylegan_052_256_10k_neut_ambcolor_pcrop_keepambcol = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_10k_neutral_pcrop_keepambcol/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.2 256\n10k int=0, Neut ambcol, tgtcol', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_031_256_30k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_30k_neutral_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB 30k sGAN v0.3.1 256\n10k int=0, Neut, pcrop', intensity=0, enable_neutral=True)

model_lab_stylegan_031_256_10k_neut_noamb_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_noprops_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, run2, pcrop', intensity=0, enable_neutral=True)
model_lab_stylegan_052_256_10k_neut_ambcolor_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.5.2_10k_neutral_ambcolor_allprops_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.2 256\n10k int=0, Neut ambcol', intensity=0, enable_neutral=True, enable_amb_color=True)
model_lab_stylegan_051_256_10k_neut_sundiam_ambcolor_pcrop_run2 = Model(outputs_path + 'model_256_lab_stylegan_0.5.1_10k_neutral_sundiam_ambcol_pcrop/23_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.1 256\n10k int=0, Neut ambcol+sundiam', intensity=0, enable_neutral=True, enable_amb_color=True, enable_sun_diam=True)
model_lab_stylegan_060_256_10k_neut_sdiam_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.6.0_10k_neutral_sundiam_new_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.6.0 256\n10k int=0, Neut, sundiam', intensity=0, enable_neutral=True, enable_sun_diam=True)
model_lab_stylegan_031_256_10k_neut_pcrop_run2 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_amb_pcrop_run2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, run2, pcrop', intensity=0, enable_neutral=True,)
model_lab_stylegan_031_256_10k_neut_facetone_pcrop = Model(outputs_path_tushar_ds3 + 'model_256_10k_lab_0.3.1_70kffhq_facetone/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, Facetone', intensity=0, enable_neutral=True, enable_face_tone=True)

model_lab_stylegan_031_256_10k_neut_bs60_pcrop_run2_ep11 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_pcrop_bs60_run2/11_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256 ep11\n10k int=0, Neut, bs60 run2', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neut_bs60_pcrop_run2_ep12 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_pcrop_bs60_run2/12_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256 ep12\n10k int=0, Neut, bs60 run2', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neut_bs60_pcrop_run2_ep13 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_pcrop_bs60_run2/13_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256 ep13\n10k int=0, Neut, bs60 run2', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neut_bs60_pcrop_run2 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_pcrop_bs60_run2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, bs60 run2', intensity=0, enable_neutral=True)

model_lab_stylegan_051_256_10k_neut_sundiam_ambcolor_pcrop_ep11 = Model(outputs_path + 'model_256_lab_stylegan_0.5.1_10k_neutral_sundiam_ambcol_pcrop/11_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.1 256\n10k int=0, Neut, ambcolor', intensity=0, enable_neutral=True, enable_sun_diam=True, enable_amb_color=True)
model_lab_stylegan_051_256_10k_neut_sundiam_ambcolor_pcrop_ep12 = Model(outputs_path + 'model_256_lab_stylegan_0.5.1_10k_neutral_sundiam_ambcol_pcrop/12_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.1 256\n10k int=0, Neut, ambcolor', intensity=0, enable_neutral=True, enable_sun_diam=True, enable_amb_color=True)
model_lab_stylegan_051_256_10k_neut_sundiam_ambcolor_pcrop_ep13 = Model(outputs_path + 'model_256_lab_stylegan_0.5.1_10k_neutral_sundiam_ambcol_pcrop/13_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.1 256\n10k int=0, Neut, ambcolor', intensity=0, enable_neutral=True, enable_sun_diam=True, enable_amb_color=True)
model_lab_stylegan_051_256_10k_neut_sundiam_ambcolor_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.5.1_10k_neutral_sundiam_ambcol_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.5.1 256\n10k int=0, Neut, ambcolor', intensity=0, enable_neutral=True, enable_sun_diam=True, enable_amb_color=True)

model_l_stylegan_031_256_10k_neut_ft21_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/21_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 21ep', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_ft22_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/22_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 22ep', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_ft15_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/15_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 15ep', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_ft17_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/17_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 17ep', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_ft20_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/20_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 20ep', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_ft23_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/23_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, ft 23ep', intensity=0, enable_neutral=True)

model_lab_stylegan_031_256_10k_neut_bs60_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_amb_pcrop_bs40/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, bs60', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neut_bs40_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_amb_pcrop_bs60/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, bs40', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neut_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_amb_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256\n10k int=0, Neut, amb', intensity=0, enable_neutral=True)
model_l_stylegan_031_256_10k_neut_pcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_neutral_amb_pcrop/14_net_G.pth', input_mode='L', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L sGAN v0.3.1 256\n10k int=0, Neut, amb', intensity=0, enable_neutral=True)
model_lab_stylegan_070_256_10k_neut_suncol_suncoltodec_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.7.0_10k_neutral_suncol_suncoltodec/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.7.0 256\n10k int=0, Neut, suncol', intensity=0, enable_neutral=True, enable_sun_color=True)
model_lab_stylegan_070_256_10k_neut_suncol_pcrop = Model(outputs_path + 'model_256_lab_stylegan_0.7.0_10k_neutral_suncol_pcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.7.0 256\n10k int=0, Neut, suncol', intensity=0, enable_neutral=True, enable_sun_color=True)

model_l_dpr_256_10k = Model(outputs_path + 'model_256_l_dpr_10k/14_net_G.pth', input_mode='L', sh_const=0.7, resolution=256, ambience=None, nc_sh=1, dataset_name='dpr', name='L DPR 256 10k\n')
model_lab_stylegan_060_256_10k_neut_sdiam_oldcrop = Model(outputs_path + 'model_256_lab_stylegan_0.6.0_10k_neutral_int_amb_sdiam_oldcrop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.6.0 256\n10k int=0, Neut', intensity=0, enable_neutral=True, enable_sun_diam=True)
model_l_3dulight_08_256_10k = Model(outputs_path + 'model_256_l_3dulab_0.8_10k/14_net_G.pth', input_mode='L', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L 3DUL v0.8 256\n10k int=0, Neut, nocrop')
model_l_stylegan_031_256_10k_paddedcrop = Model(outputs_path + 'model_256_l_stylegan_0.3.1_10k_int_amb_paddedcrop/14_net_G.pth', input_mode='L', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L sGAN v0.3.1 256\n10k int=0, pcrop', intensity=0)

model_l_stylegan_021_256_10k_neut = Model(outputs_path + 'model_256_l_stylegan_0.2.1_20k_nocrop_neutral_int_amb/14_net_G.pth', input_mode='L', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L sGAN v0.2.1 256\n10k int=0, Neut, nocrop', intensity=0, enable_neutral=True)
model_lab_stylegan_021_256_10k_neut_run2 = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_nocrop_neutral_int_amb_run1/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neut, Run1', intensity=0, enable_neutral=True)
model_lab_stylegan_021_256_10k_neut_run1 = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_nocrop_neutral_int_amb_run2/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neut, Run2', intensity=0, enable_neutral=True)
# model_lab_stylegan_021_256_10k_neutral_adec = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_20k_nocrop_neutral_int_amb_crop_shafeatdec/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neut, a to dec', intensity=0, model_neutral=True)
# model_lab_stylegan_021_256_10k_neutral_shafeatdec = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_20k_nocrop_neutral_int_amb_crop_shafeatdec/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neut, sh+a to dec', intensity=0, model_neutral=True)

#All models above have intensity and ambience input if not otherwise noted, their variation depends on the dataset version
#All above models have SHFIX3

# model_lab_stylegan_021_256_10k_neutral_sgandec = Model(outputs_path_tushar + 'model_256_lab_stylegan_0.2.1_10k_nocrop_desc_last10k_Stylegan/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neutral, sgan real', intensity=0, model_neutral=True)
model_lab_stylegan_021_256_10k_neutral_sgan_ffhq_dec = Model(outputs_path + 'model_256_lab_0.2.1_nocrop_sgan_ffhq/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256\n10k int=0, Neutral, sgan+ffhq real', intensity=0, enable_neutral=True)
# # '/home/nedko/face_relight/outputs/model_256_lab_0.2.1_nocrop_sgan_ffhq'

model_lab_stylegan_021_256_20k_neutral_intensity_shfix3 = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_20k_nocrop_neutral_int_amb_crop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n20k int=0, Neutral, SHFIX3', intensity=0, enable_neutral=True)
model_lab_stylegan_031_256_10k_neutral_crop_intensity_ambient_shfix3 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_neutral_int_amb_crop/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, intensity=0, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256 \n10k int=0, Neutral, SHFIX3', enable_neutral=True)
model_lab_stylegan_031_256_10k_crop_intensity_ambient_shfix3 = Model(outputs_path + 'model_256_lab_stylegan_0.3.1_10k_intensity_ambient_crop_shfix3/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3.1 256 \n10k int=0, Neutral, crop, SHFIX3', intensity=0)
model_lab_stylegan_02_256_10k_neutral_intensity_shfix3 = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k_neutral_int_amb_shfix3/14_net_G.pth', input_mode='LAB', resolution=256, ambience=None, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k int=0, Neutral, SHFIX3', intensity=0, enable_neutral=True)
model_lab_stylegan_021_256_10k_nocrop_intensity_ambient_shfix3 = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_nocrop_intensity_ambient_shfix3/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, nc_sh=1, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256 \n10k int=0, amb=c, SHFIX3', intensity=0)
model_lab_stylegan_03_256_10k_nocrop_intensity_ambient_0 = Model(outputs_path + 'model_256_lab_stylegan_0.3_10k_nocrop_intensity_ambient/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.080, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3 256 \n10k int=0, amb=0.08', intensity=0)
model_lab_stylegan_03_256_10k_nocrop_intensity_ambient_1 = Model(outputs_path + 'model_256_lab_stylegan_0.3_10k_nocrop_intensity_ambient/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.180, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3 256 \n10k int=0, amb=0.18', intensity=0)
model_lab_stylegan_03_256_10k_nocrop_intensity_ambient_2 = Model(outputs_path + 'model_256_lab_stylegan_0.3_10k_nocrop_intensity_ambient/14_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.3 256 \n10k int=0, amb=0.28', intensity=0)
model_lab_stylegan_021_256_10k_nocrop_intensity_ambient = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_nocrop_intensity_ambient/10_net_G.pth', input_mode='LAB', resolution=256, ambience=0.280, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256 \n10k int=0, amb=c, 10ep', intensity=0)

model_lab_stylegan_021_256_10k_nocrop_intensity = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_nocrop_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256 \n10k intensity=0, not cropped', intensity=0)
model_lab_stylegan_021_256_10k_intensity = Model(outputs_path + 'model_256_lab_stylegan_0.2.1_10k_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2.1 256 \n10k intensity=0, cropped', intensity=0)
model_lab_3dulight_08_256_10k_intensity = Model(outputs_path + 'model_256_lab_3dulight_0.8_10k_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB 3DULight v0.8 256 \n10k Blender intensity')
model_lab_stylegan_01_256_10k_intensity = Model(outputs_path + 'model_256_lab_stylegan_0.1_10k_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.1 256 \n10k Blender intensity')
model_lab_3dul_08_neutral_256_10k_intensity = Model(outputs_path + 'model_256_lab_3dulight_0.8_10k_intensity_neutral/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB 3DUL v0.8 256 Neutral \n10k Blender intensity', enable_neutral=True)
model_lab_stylegan_02_neutral_256_10k_intensity_ft = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k_intensity/19_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB FT sGAN v0.2 256 Neutral \n10k Blender intensity', enable_neutral=True)
model_lab_stylegan_01_neutral_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_neutral/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.1 256 Neutral \n10k Blender intensity', enable_neutral=True)
model_lab_stylegan_04_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.4_10k/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.4 256 \n10k Blender intensity')
model_lab_stylegan_02_256_10k_intensity_debug = Model(outputs_path + 'model_256_labfull_stylegan_0.1_10k_debug/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 \n10k Blender intensity')
model_lab_stylegan_02_neutral_256_10k_intensity = Model(outputs_path + 'model_256_labfull_stylegan_0.2_10k_intensity/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.2 256 Neutral \n10k Blender intensity', enable_neutral=True)
model_l_stylegan_02_neutral_256_10k_intensity = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k_intensity/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.2 256 Neutral \n10k Blender intensity', enable_neutral=True)
model_l_stylegan_03_neutral_256_10k = Model(outputs_path + 'model_256_lab_stylegan_0.3_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.3 256 Neutral 10k', enable_neutral=True)
model_l_styleganw_02_neutral_256_10k = Model(outputs_path + 'model_256_lab_stylegan_0.2_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.2 256 Neutral 10k', enable_neutral=True)
model_lab_stylegan_01_256_10k = Model(outputs_remote_path + 'model_256_lab_3dulab_0.8_10k_l+ab/14_net_G.pth', input_mode='LAB', resolution=256, dataset_name='3dulight_shfix2', name='L+AB sGAN v0.1 256 10k')
model_l_stylegan_01_256_10k_ep1fix = Model(outputs_path + 'model_256_lab_stylegan_0.1_10k_ep1fix/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 10k ep1fix')

model_l_stylegan_01_neutral_256_10k = Model(outputs_remote_path + 'model_256_lab_neutral_sgan_0.1_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 Neutral 10k', enable_neutral=True)
model_l_stylegan_01_256_10k = Model(outputs_path + 'model_256_lab_sgan_0.1_10k/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 256 10k')
model_l_3dulight_08_256_10k_shfix2_test = Model(outputs_path + 'model_256_lab_3dulab_0.8_test_10k_2/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 10k')
# model_l_3dulight_08_256_10k_shfix2_test = Model(outputs_path + 'model_256_lab_3dulab_0.8_test_10k/13_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 10k New')
model_l_3dulight_08_256_full_shfix2 = Model(outputs_path + 'model_256_lab_3dulab_0.8_test/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L 3DUL 256 30k')
model_l_stylegan_01_256_neutral_full = Model(outputs_path + 'model_neutral_256_lab_stylegan_v0.1/14_net_G.pth', input_mode='L', resolution=256, dataset_name='3dulight_shfix2', name='L sGAN v0.1 Neutral 256 30k', enable_neutral=True)
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
    # model_lab_stylegan_031_256_10k_neut_pcrop,
    # model_lab_stylegan_031_256_20k_neut_pcrop,
    # model_lab_stylegan_031_256_20k_neut_pcrop_rmix30k,
    # model_lab_stylegan_031_256_30k_neut_pcrop,
    # model_lab_stylegan_031_256_30k_neut_pcrop_rmix30k,
    # model_lab_stylegan_031_256_50k_neut_pcrop,
    # model_lab_stylegan_031_256_70k_neut_pcrop,
    # model_lab_stylegan_031_256_70k_neut_pcrop_rmix70k,

    # model_lab_stylegan_070_256_10k_neut_suncol_pcrop,
    # model_lab_stylegan_052_256_10k_neut_ambcolor_pcrop,
    # model_lab_stylegan_052_256_10k_neut_ambcolor_pcrop_keepambcol,
    # model_lab_stylegan_081_256_10k_neut_ambcolor_suncolor_pcrop,
    # model_lab_stylegan_052_256_20k_neut_pcrop,
    # model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k,
    # model_lab_stylegan_052_256_30k_neut_pcrop,
    # model_lab_stylegan_052_256_30k_neut_pcrop_rmix30k,
    # model_lab_stylegan_052_256_70k_neut_pcrop_rmix70k
    # model_512_lab_stylegan_052_20k_r30k_30k_neutral_pcrop,

    # model_lab_stylegan_060_256_10k_neut_sdiam_oldcrop,
    # model_lab_stylegan_060_256_10k_neut_sdiam_pcrop,

    # model_lab_stylegan_031_256_10k_neut_facetone_pcrop

    # model_l_stylegan_031_256_10k_neut_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft15_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft17_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft20_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft21_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft22_pcrop,
    # model_lab_stylegan_031_256_10k_neut_ft23_pcrop

    # model_lab_stylegan_031_256_10k_neut_bs40_pcrop,
    # model_lab_stylegan_031_256_10k_neut_bs60_pcrop,
    # model_l_stylegan_031_256_10k_paddedcrop,
    # model_l_dpr_256_10k,

    #     G_D comparison models
    # model_256_lab_stylegan_031_50k_neutral_pcrop_r30k_r30k,
    model_256_lab_stylegan_031_50k_neutral_pcrop_r30k_r30k_rerun,
    # model_256_lab_stylegan_031_50k_neutral_pcrop_r50k_r50k,
    # model_256_lab_stylegan_031_50k_neutral_pcrop_r50k_r50k_rerun

#compare ft
# model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k, model_lab_stylegan_052_256_20k_neut_pcrop_rmix30k_ft

#     overfit
# model_256_lab_stylegan_031_100_neutral_pcrop_r30k_r30k_overfit,

    # rgb
model_256_rgb_stylegan_031_50k_neutral_pcrop_r30k_r0k
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
    # result = cv2.resize(out_img, (col, row))
    result = resize_pil(out_img, col, row)

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

    # out_img = cv2.resize(out_img, (crop_sz[1], crop_sz[0]))
    out_img = resize_pil(out_img, crop_sz[1], crop_sz[0])



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



def load_model(checkpoint_dir_cmd, device, input_mode='L', model_1024=False, model_neutral=False, enable_ambient=False,
               nc_sh=1, enable_sun_diam=False, enable_sun_color=False, enable_amb_color=False, enable_face_tone=False):
    if input_mode in ['L', 'LAB','RGB']:
        nc_img = 3 if input_mode == 'LAB' or input_mode == 'RGB'  else 1
        if model_1024:
            my_network_512 = HourglassNet_512_1024(16)
            my_network = HourglassNet_1024(my_network_512, 16)
        else:
            nc_light_extra=0

            if enable_sun_color:
                nc_light_extra += 3

            if enable_amb_color:
                nc_light_extra += 3

            if enable_sun_diam:
                nc_light_extra += 1

            if enable_ambient:
                nc_light_extra += 1

            if enable_face_tone:
                nc_light_extra += 2

            my_network = HourglassNet(enable_target=not model_neutral, ncImg=nc_img, ncLightExtra=nc_light_extra, ncLight=9*nc_sh)
    else:
        my_network = HourglassNet_RGB()

    current_state_dict = [module.state_dict().keys() for module in my_network.modules()]
    loaded_params = torch.load(checkpoint_dir_cmd)
    my_network.load_state_dict(loaded_params)
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
    # sh = sh[0:9]
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

    return output_img, sh, output_sh.cpu().detach().numpy()


def get_output_params(output_params_tensor, model_obj):
    '''
    Convert output prediction tensor to organized dictionary of predictions
    :param output_params_tensor: The predictions tensor from the encoder
    :param model_obj: the model object
    :return: a dictionary with organized parameters
    '''
    output_params = {}
    output_params_tensor = output_params_tensor.flatten()

    if len(output_params_tensor) >= 12 and model_obj.enable_amb_color:
        output_params['ambient_color'] = output_params_tensor[-3:]

    return output_params

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
    model_obj.model = load_model(model_obj.checkpoint_path, model_obj.device, input_mode=model_obj.input_mode,
                                 model_1024=model_obj.model_1024, model_neutral=model_obj.model_neutral,
                                 enable_ambient=model_obj.ambience is not None, nc_sh=model_obj.nc_sh, enable_sun_diam=model_obj.sun_diam,
                                 enable_sun_color=model_obj.enable_sun_color, enable_amb_color=model_obj.enable_amb_color,
                                 enable_face_tone=model_obj.enable_face_tone)

input_processor = InputProcessor(model_dir, enable_segment, device)

for orig_path, out_fname, gt_data in dataset.iterate():

    sh_path_dataset = None
    gt_path = None
    max_res_deviation = 300

    if gt_data is not None:
        gt_path, sh_path_dataset = gt_data

    orig_img = cv2.imread(orig_path)

    orig_img_proc, mask, loc, crop_sz, border = input_processor.process_side_offset(orig_img)
    is_face_found = orig_img is not None

    # orig_img_proc = resize_pil(orig_img_proc, height=min_resolution)

    if not is_face_found:
        print('No face found: ', orig_path)
        continue

    if enable_test_mode:
        results = []

        if enable_face_boxes:
            orig_img_show = orig_img_proc
        else:
            orig_img_show = orig_img

        if orig_img_show.shape[0] < min_resolution:
            continue

        ##UNCOMMENT FOR ORIGINAL IMAGE EXPORT
        # if orig_img_show.shape[0] > min_resolution:
        #     orig_img_show = resize_pil(orig_img_show, height=min_resolution)

            # results_frame.append(result_img)
        # results = [orig_img_show]
    else:
        if enable_face_boxes:
            orig_img_show = orig_img_proc
        else:
            orig_img_show = orig_img

        if orig_img_show.shape[0] < min_resolution:
            continue

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
        results_params = []
        for model_obj in model_objs:

            if sh_path_dataset is None:
                sh_path = model_obj.sh_path
                target_sh = model_obj.target_sh[sh_idx]
                sh_fname = None
            else:
                sh_path, sh_fname = sh_path_dataset.rsplit('/', 1)
                target_sh = None

            extra_ops={}


            result_img, sh, output_params_tensor = test(model_obj, orig_img_proc, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=target_sh, sh_path=sh_path, sh_fname=sh_fname, extra_ops=extra_ops, intensity=model_obj.intensity)

            output_params = get_output_params(output_params_tensor, model_obj)

            result_img = handle_output(result_img, orig_img_proc.shape[1], orig_img_proc.shape[0], mask, orig_img_proc, orig_img, loc, crop_sz, border, enable_face_boxes, orig_path.rsplit('/', 1)[-1].rsplit('.', 1)[0], target_sh, sh, model_obj.blend_mode)

            if result_img.shape[0]>min_resolution:
                result_img = resize_pil(result_img, height=min_resolution)

            result_img = np.ascontiguousarray(result_img, dtype=np.uint8)

            if not enable_test_mode:
                result_img = draw_label(result_img, model_obj.name)

            ##COMMENT FOR ORIGINAL IMAGE EXPORT
            results_frame.append(result_img)

            results_params.append(output_params)
        # tgt_result = cv2.resize(tgt_result, (256,256))

        params_path = osp.join(out_dir, out_fname.rsplit('.', 1)[0] + '.txt')
        # Todo: save txt ??
        # np.savetxt(params_path, np.array(results_params[0]['ambient_color']))

        results_params = np.array(results_params)
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