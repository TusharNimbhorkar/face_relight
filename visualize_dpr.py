import argparse
import os.path as osp
import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from commons.common_tools import FileOutput
from utils.utils_SH import get_shading
from models.skeleton512_rgb import HourglassNet as HourglassNet_RGB
from models.skeleton512 import HourglassNet
# for 1024 skeleton
from models.skeleton1024 import HourglassNet as HourglassNet_512_1024
from models.skeleton1024 import HourglassNet_1024

from PIL import  Image
import torch

# for segmentation
from demo_segment.model import BiSeNet
import torchvision.transforms as transforms


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default='test_data/test_images', required=False,
	help="Input Directory")
ap.add_argument("-o", "--output", default='outputs/test', required=False,
	help="Output Directory")
ap.add_argument("-d", "--device", default='cuda', required=False, choices=['cuda','cpu'],
	help="Device")
args = vars(ap.parse_args())

device = args['device']

lightFolder_dpr = 'test_data/00/'
lightFolder_3dulight_shfix = 'test_data/sh_presets/horizontal'
lightFolder_3dulight = 'test_data/sh_presets/horizontal_old'
out_dir = args["output"]#'/home/nedko/face_relight/dbs/comparison'

target_sh_id_dpr = list(range(72)) + [71]*20#60#5 #60
target_sh_id_3dulight = list(range(90))# 70 # 19#89
target_sh_id_3dulight_shfix = list(range(90))#75 # 19#89

min_video_frames = 10
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
    def __init__(self, checkpoint_path, lab, resolution, dataset_name, sh_const=1.0, name='',model_1024=False):
        self.checkpoint_path = checkpoint_path
        self.lab = lab
        self.resolution = resolution
        self.model = None
        self.sh_const = sh_const
        self.name=name
        self.device = device ##TODO
        self.model_1024=model_1024
        # self.post_flag = post_flag


        if dataset_name == 'dpr':
            self.sh_path = lightFolder_dpr
            self.target_sh = target_sh_id_dpr
        elif dataset_name == '3dulight':
            self.sh_path = lightFolder_3dulight
            self.target_sh = target_sh_id_3dulight
        elif dataset_name == '3dulight_shfix':
            self.sh_path = lightFolder_3dulight_shfix
            self.target_sh = target_sh_id_3dulight_shfix

    def __call__(self, input_img, target_sh, *args, **kwargs):
        target_sh = torch.autograd.Variable(torch.from_numpy(target_sh).to(self.device))

        if self.lab:
            Lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
            input_img = Lab[:, :, 0]
            input_img = input_img.astype(np.float32) / 255.0
            input_img = input_img.transpose((0, 1))
            input_img = input_img[None, None, ...]
        else:
            input_img = input_img
            input_img = input_img.astype(np.float32)
            input_img = input_img / 255.0
            input_img = input_img.transpose((2, 0, 1))
            input_img = input_img[None, ...]

        torch_input_img = torch.autograd.Variable(torch.from_numpy(input_img).to(self.device))

        model_output = self.model(torch_input_img, target_sh, *args)
        output_img, _, outputSH, _ = model_output

        output_img = output_img[0].cpu().data.numpy()
        output_img = output_img.transpose((1, 2, 0))
        output_img = np.squeeze(output_img)  # *1.45
        output_img = (output_img * 255.0).astype(np.uint8)

        if self.lab:
            Lab[:, :, 0] = output_img
            output_img = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        else:
            output_img = output_img

        return output_img, outputSH

class ModelSegment(Model):
    def __init__(self, checkpoint_path, lab, resolution, dataset_name, sh_const=1.0, name='',post_flag=False):
        super(ModelSegment,self).__init__(checkpoint_path, lab, resolution, dataset_name, sh_const, name)
        self.post_flag = post_flag

    def __call__(self, input_img, target_sh, *args, **kwargs):
        output_img, output_sh = super().__call__(input_img, target_sh, *args, **kwargs)

        rgb_img_path = kwargs['orig_img']

        rgb_img = Image.open(osp.join(rgb_img_path))
        segment = evaluate(rgb_img,self.post_flag, self.device)
        rgb_img = rgb_img.resize((input_img.shape[0], input_img.shape[1]), resample=Image.LANCZOS)
        segment[segment == 255] = 1
        segment = np.array(Image.fromarray(segment).resize((input_img.shape[0], input_img.shape[0]), resample=Image.LANCZOS))

        seg_img = np.zeros_like(output_img)
        seg_img[:, :, 0] = segment
        seg_img[:, :, 1] = segment
        seg_img[:, :, 2] = segment
        # seg_img = np.array(Image.fromarray(seg_img).resize((res, res), resample=Image.LANCZOS))
        output_img = np.multiply(output_img, seg_img)

        mask = seg_img.copy()
        background = np.copy(rgb_img)
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        foreground = np.copy(output_img)
        mask = mask.astype(float)
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(mask, foreground)

        # Multiply the background with ( 1 - alpha )
        try:
            background = cv2.multiply(1 - mask, background)
        except:
            print(background.shape, mask.shape)

        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        # old
        # output_img = np.multiply(output_img, seg_img)


        return outImage, output_sh

class ModelSegment_blend(Model):
    def __init__(self, checkpoint_path, lab, resolution, dataset_name, sh_const=1.0, name='',post_flag=False):
        super(ModelSegment_blend,self).__init__(checkpoint_path, lab, resolution, dataset_name, sh_const=1.0, name='')
        self.post_flag = post_flag

    def __call__(self, input_img, target_sh, *args, **kwargs):
        output_img, output_sh = super().__call__(input_img, target_sh, *args, **kwargs)

        rgb_img_path = kwargs['orig_img']

        rgb_img = Image.open(osp.join(rgb_img_path))
        segment = evaluate(rgb_img,face=self.post_flag)

        rgb_img =  rgb_img.resize((input_img.shape[0], input_img.shape[1]), resample=Image.LANCZOS)
        segment = np.array(
            Image.fromarray(segment).resize((rgb_img.size[0], rgb_img.size[1]), resample=Image.LANCZOS))

        vis_parsing_anno = segment.copy()
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

        alpha_2 = np.zeros(shape=(vis_parsing_anno.shape[0],vis_parsing_anno.shape[1],3))
        alpha_2[:,:,0] = np.copy(vis_parsing_anno)
        alpha_2[:,:,1] = np.copy(vis_parsing_anno)
        alpha_2[:,:,2] = np.copy(vis_parsing_anno)
        kernel = np.ones((10, 10), np.uint8)
        alpha_2 = cv2.erode(alpha_2, kernel, iterations=1)
        alpha_2 = cv2.GaussianBlur(alpha_2,(29,29),15,15)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha_2 = alpha_2.astype(float) / 255

        mask = alpha_2.copy()
        background = np.copy(rgb_img)
        background = cv2.cvtColor(background,cv2.COLOR_RGB2BGR)
        foreground = np.copy(output_img)
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(mask, foreground)

        # Multiply the background with ( 1 - alpha )
        try:
            background = cv2.multiply(1 - mask, background)
        except:
            print(background.shape,mask.shape)

        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        return outImage, output_sh

# dataset_test = DatasetDefault('path/to/files')
dataset_3dulight_v0p8 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.8_256/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p7_randfix =  Dataset3DULightGT('/home/tushar/data2/face_relight/dbs/3dulight_v0.7_256_fix/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p6 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.6_256/train', n_samples=5, n_samples_offset=5) # DatasetDPR('/home/tushar/data2/DPR/train')

model_lab_3dulight_08_bs7 = Model('/home/nedko/face_relight/outputs/remote/outputs/model_256_lab_3dulight_v0.8_full_bs7/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DUL v0.8 30k bs7')
model_lab_3dulight_08_seg_face = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_seg_face/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 10k Segment')
model_lab_dpr_seg = Model('/home/tushar/data2/checkpoints_debug/model_fulltrain_dpr7_mse_sumBS20_ogsegment/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', sh_const = 0.7, name='LAB DPR v0.8 10k Segment')
model_lab_3dulight_08_seg = Model('/home/tushar/data2/face_relight/outputs/model_256_3du_v08_lab_seg/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 10k Segment')
model_lab_3dulight_08_full_seg = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_full/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 30k Segment +hair')
model_lab_3dulight_08_bs16 = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_bs16//14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 bs16')
model_rgb_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v0.8_full/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight_shfix', name='RGB 3DULight v0.8 30k')
model_rgb_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight_shfix', name='RGB 3DULight v0.8')
model_lab_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8_full/model_256_lab_3dulight_v0.8_full/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight_shfix', name='LAB 3DULight v0.8 30k')
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
model_lab_dpr_10k = Model('/home/tushar/data2/checkpoints/model_256_dprdata10k_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', sh_const = 0.7, name='LAB DPR 10K')
model_lab_pretrained = Model('models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'

model_lab_pretrained1 = ModelSegment_blend('models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR',post_flag =True ) # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'
model_lab_pretrained2 = ModelSegment('models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR',post_flag =True ) # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'

model_lab_3dulight_1024_one_third = Model('/home/tushar/FR/face_relight/models/model_256_3dudata/v08_1024_third/12_net_G.pth', lab=True, model_1024 =True,resolution=1024, dataset_name='3dulight_shfix', sh_const = 1.0, name='1024 3dulight 10k') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'


model_objs = [
    # model_lab_3dulight_08_bs16,
    # model_lab_3dulight_08_bs7,
    # model_lab_3dulight_08_full
    model_lab_3dulight_1024_one_third,
model_lab_pretrained2
    # model_lab_dpr_seg
]

# dataset = dataset_3dulight_v0p8
dataset = DatasetDefault(args["input"])

# checkpoint_src = '/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.3/model_256_lab_3dulight_v0.3/14_net_G.pth'
# checkpoint_tgt = '/home/tushar/data2/checkpoints/model_256_3dudataset_lab/model_256_3dudataset_lab/14_net_G.pth' #'/home/tushar/data2/checkpoints/face_relight/outputs/model_rgb_light3du/14_net_G.pth' #'/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'



# Functions for segment parsing

def vis_parsing_maps(im, parsing_anno, stride, h=None, w=None,face=False):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # MASK
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (w, h))

    # only face skin
    vis_parsing_anno[vis_parsing_anno == 16] = 0
    vis_parsing_anno[vis_parsing_anno == 14] = 0
    vis_parsing_anno[vis_parsing_anno == 7] = 0
    vis_parsing_anno[vis_parsing_anno == 8] = 0
    if face:
        vis_parsing_anno[vis_parsing_anno==17]=0

    vis_parsing_anno[vis_parsing_anno > 0] = 255


    return vis_parsing_anno


def evaluate(img,face,device):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    save_pth = osp.join('demo_segment/cp', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        w,h = img.size
        # h = img.shape[1]
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        output_img = vis_parsing_maps(image, parsing, stride=1, h=h, w=w,face=face)
    return output_img



def load_model(checkpoint_dir_cmd, device, lab=True, model_1024=False):
    if lab:
        if model_1024:
            my_network_512 = HourglassNet_512_1024(16)
            my_network = HourglassNet_1024(my_network_512, 16)
        else:
            my_network = HourglassNet()
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

def test(my_network, input_img, lab=True, sh_id=0, sh_constant=1.0, res=256, sh_path=lightFolder_3dulight, sh_fname=None, extra_ops={}):
    img = input_img
    row, col, _ = img.shape
    # img = cv2.resize(img, size_re)
    img = np.array(Image.fromarray(img).resize((res, res), resample=Image.LANCZOS))
    # cv2.imwrite('1.png',img)


    if sh_fname is None:
        sh_fname = 'rotate_light_{:02d}.txt'.format(sh_id)

    sh = np.loadtxt(osp.join(sh_path, sh_fname))
    sh = sh[0:9]
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


    return output_img


for model_obj in model_objs:
    model_obj.model = load_model(model_obj.checkpoint_path, model_obj.device, lab=model_obj.lab, model_1024=model_obj.model_1024)

for orig_path, out_fname, gt_data in dataset.iterate():

    video_out = FileOutput(osp.join(out_dir, out_fname.rsplit('.',1)[0]+'.mp4'))
    sh_path_dataset = None
    gt_path = None

    if gt_data is not None:
        gt_path, sh_path_dataset = gt_data

    # orig_path = osp.join(dir, dir.rsplit('/',1)[-1] + '_05.png')
    # left_path = osp.join(dir, dir.rsplit('/', 1)[-1] + '_00.png')
    orig_img = cv2.imread(orig_path)
    # left_img = cv2.imread(left_path)

    if orig_img.shape[0] > min_resolution:
        # orig_img = cv2.resize(orig_img, (min_resolution,min_resolution))
        orig_img = np.array(Image.fromarray(orig_img).resize((min_resolution,min_resolution), resample=Image.LANCZOS))

    results = [orig_img]

    if gt_path is not None:
        gt_img = cv2.imread(gt_path)

        if gt_img.shape[0] > min_resolution:
            # gt_img = cv2.resize(gt_img, (min_resolution, min_resolution))
            gt_img = np.array(
                Image.fromarray(gt_img).resize((min_resolution, min_resolution), resample=Image.LANCZOS))

        cv2.putText(gt_img, 'Ground Truth', (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
        results.append(gt_img)

    if sh_path_dataset is None:
        min_sh_list_len = min([len(model_obj.target_sh) for model_obj in model_objs])
    else:
        min_sh_list_len = 1

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
            if isinstance(model_obj, ModelSegment) or isinstance(model_obj, ModelSegment_blend):
                extra_ops['orig_img'] = orig_path

            result_img = test(model_obj, orig_img, lab=model_obj.lab, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=target_sh, sh_path=sh_path, sh_fname=sh_fname, extra_ops=extra_ops)

            if result_img.shape[0]>min_resolution:
                # result_img = cv2.resize(result_img, (min_resolution,min_resolution))
                result_img = np.array(Image.fromarray(result_img.astype(np.uint8)).resize((min_resolution, min_resolution), resample=Image.LANCZOS))


            result_img = np.ascontiguousarray(result_img, dtype=np.uint8)
            cv2.putText(result_img, model_obj.name, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
            results_frame.append(result_img)

        # tgt_result = cv2.resize(tgt_result, (256,256))

        out_img = np.concatenate(results + results_frame, axis=1)
        print(orig_path, gt_path)

        if min_sh_list_len > min_video_frames:
            video_out.post(out_img)
        else:
            cv2.imwrite(osp.join(out_dir, out_fname.rsplit('.', 1)[0] + '_' + str(sh_idx) + '.' + out_fname.rsplit('.', 1)[1]), out_img)


    video_out.close()
    # plt.imshow(out_img[:,:,::-1])
    # plt.show()