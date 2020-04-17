import argparse
import os.path as osp
import glob
import cv2
import imutils
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
import dlib

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
enable_forced_image_out = args['force_images']
enable_segment = args['segment']
enable_face_boxes = args['crops_only']
enable_test_mode = args['test']

lightFolder_dpr = 'test_data/00/'
lightFolder_3dulight = 'test_data/sh_presets/horizontal'
model_dir = osp.abspath('./demo/data/model')
out_dir = args["output"]#'/home/nedko/face_relight/dbs/comparison'

target_sh_id_dpr = list(range(72)) + [71]*20#60#5 #60
target_sh_id_3dulight = list(range(90 - 22 - 45, 90 - 22 + 1))#75 # 19#89

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
        elif dataset_name == '3dulight':
            self.sh_path = lightFolder_3dulight
            self.target_sh = target_sh_id_3dulight

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

# dataset_test = DatasetDefault('path/to/files')
dataset_3dulight_v0p8 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.8_256/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p7_randfix =  Dataset3DULightGT('/home/tushar/data2/face_relight/dbs/3dulight_v0.7_256_fix/train', n_samples=5, n_samples_offset=0) # DatasetDPR('/home/tushar/data2/DPR/train')
dataset_3dulight_v0p6 = Dataset3DULightGT('/home/nedko/face_relight/dbs/3dulight_v0.6_256/train', n_samples=5, n_samples_offset=5) # DatasetDPR('/home/tushar/data2/DPR/train')

model_lab_3dulight_08_1024_10k = Model('/home/tushar/data2/face_relight/outputs/model_1024_3du_v08_lab_third/14_net_G.pth', lab=True, resolution=1024, dataset_name='3dulight', name='LAB 3DUL v0.8 1024 10k', model_1024=True)
model_lab_3dulight_08_bs7 = Model('/home/nedko/face_relight/outputs/remote/outputs/model_256_lab_3dulight_v0.8_full_bs7/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DUL v0.8 30k bs7')
model_lab_3dulight_08_seg_face = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_seg_face/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8 10k Segment')
model_lab_dpr_seg = Model('/home/tushar/data2/checkpoints_debug/model_fulltrain_dpr7_mse_sumBS20_ogsegment/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', sh_const = 0.7, name='LAB DPR v0.8 10k Segment')
model_lab_3dulight_08_seg = Model('/home/tushar/data2/face_relight/outputs/model_256_3du_v08_lab_seg/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8 10k Segment')
model_lab_3dulight_08_full_seg = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_full/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8 30k Segment +hair')
model_lab_3dulight_08_bs16 = Model('/home/nedko/face_relight/outputs/remote/outputs/3dulight_v0.8_256_bs16//14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8 bs16')
model_rgb_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v0.8_full/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight', name='RGB 3DULight v0.8 30k')
model_rgb_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_rgb_3dulight_v0.8/model_256_rgb_3dulight_v08_rgb/14_net_G.pth', lab=False, resolution=256, dataset_name='3dulight', name='RGB 3DULight v0.8')
model_lab_3dulight_08_full = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8_full/model_256_lab_3dulight_v0.8_full/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8 30k')
model_lab_3dulight_08 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.8/model_256_lab_3dulight_v0.8/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.8')
model_lab_3dulight_07_randfix = Model('/home/tushar/data2/face_relight/outputs/model_256_lab_3dulight_v0.7_random_ns5/model_256_lab_3dulight_v0.7_random_ns5/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.7 RANDFIX')
model_lab_3dulight_07 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.7_dlfix_ns15/model_256_lab_3dulight_v0.7_dlfix_ns15/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.7')
model_lab_3dulight_06 = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.6_dlfix_ns15/model_256_lab_3dulight_v0.6_dlfix_ns15/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.6')
model_lab_3dulight_05_shfix = Model('/home/nedko/face_relight/outputs/model_256_lab_3dulight_v0.5_shfix/model_256_lab_3dulight_v0.5_shfix/14_net_G.pth', lab=True, resolution=256, dataset_name='3dulight', name='LAB 3DULight v0.5 SHFIX')
model_lab_dpr_10k = Model('/home/tushar/data2/checkpoints/model_256_dprdata10k_lab/14_net_G.pth', lab=True, resolution=256, dataset_name='dpr', sh_const = 0.7, name='LAB DPR 10K')
model_lab_pretrained = Model('models/trained/trained_model_03.t7', lab=True, resolution=512, dataset_name='dpr', sh_const = 0.7, name='Pretrained DPR') # '/home/tushar/data2/DPR_test/trained_model/trained_model_03.t7'

model_objs = [
    # model_lab_3dulight_08_bs16,
    # model_lab_3dulight_08_bs7,
    model_lab_3dulight_08_full,
    model_lab_3dulight_08_1024_10k
    # model_lab_dpr_seg
]

# dataset = dataset_3dulight_v0p8
dataset = DatasetDefault(args["input"])

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


def handle_output(outputImg, col, row, mask, img_p, img_orig, loc, crop_sz, border, enable_face_boxes, item_name, idx):
    render_data_dir = '/home/nedko/face_relight/dbs/rendering'
    model_data_dir = '/home/nedko/face_relight/outputs/test_bg'
    out_dir = '/home/nedko/face_relight/outputs/test_bg_replace'
    masks_fname = 'mask_full.png'
    norms_fname = 'normals_diffused.png'

    # mask_path = osp.join(render_data_dir, item_name, masks_fname)
    # norms_path = osp.join(render_data_dir, item_name, norms_fname)
    # render_path = osp.join(render_data_dir, item_name, '%04d.jpg' % idx)
    # print(mask_path, norms_path)
    # mask_full = (cv2.imread(mask_path) / 255).astype(np.uint8)
    # norms = cv2.imread(norms_path)
    # render_img = cv2.imread(render_path)

    # mask = (mask/255.0)[:,:, np.newaxis]
    result = cv2.resize(outputImg, (col, row))

    # do something here
    # make a gauss blur

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TODO add original image which will be background  CHECK AGAIN  !!!!!!!!!!!

    if mask is not None:
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

# # Functions for segment parsing
# def vis_parsing_maps(im, parsing_anno, stride, h=None, w=None,face=False):
#     im = np.array(im)
#     vis_im = im.copy().astype(np.uint8)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
#     num_of_class = np.max(vis_parsing_anno)
#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
#     # MASK
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, (w, h))
#
#     # only face skin
#     vis_parsing_anno[vis_parsing_anno == 16] = 0
#     vis_parsing_anno[vis_parsing_anno == 14] = 0
#     vis_parsing_anno[vis_parsing_anno == 7] = 0
#     vis_parsing_anno[vis_parsing_anno == 8] = 0
#     if face:
#         vis_parsing_anno[vis_parsing_anno==17]=0
#
#     vis_parsing_anno[vis_parsing_anno > 0] = 255
#
#
#     return vis_parsing_anno

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

    sh_path_dataset = None
    gt_path = None

    if gt_data is not None:
        gt_path, sh_path_dataset = gt_data

    orig_img = cv2.imread(orig_path)

    orig_img_proc, mask, loc, crop_sz, border = preprocess(orig_img, device, enable_segment)
    is_face_found = orig_img is not None

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

        results = [orig_img_show]

    if gt_path is not None:
        gt_img = cv2.imread(gt_path)

        if gt_img.shape[0] > min_resolution:
            gt_img = resize_pil(gt_img, height=min_resolution)

        if not enable_test_mode:
            cv2.putText(gt_img, 'Ground Truth', (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
        results.append(gt_img)

    if sh_path_dataset is None:
        min_sh_list_len = min([len(model_obj.target_sh) for model_obj in model_objs])
    else:
        min_sh_list_len = 1

    enable_video_out = min_sh_list_len > min_video_frames and not enable_forced_image_out

    if enable_video_out:
        video_out = FileOutput(osp.join(out_dir, out_fname.rsplit('.',1)[0]+'.mp4'))

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

            result_img = test(model_obj, orig_img_proc, lab=model_obj.lab, sh_constant=model_obj.sh_const, res=model_obj.resolution, sh_id=target_sh, sh_path=sh_path, sh_fname=sh_fname, extra_ops=extra_ops)

            result_img = handle_output(result_img, orig_img_proc.shape[1], orig_img_proc.shape[0], mask, orig_img_proc, orig_img, loc, crop_sz, border, enable_face_boxes, orig_path.rsplit('/', 1)[-1].rsplit('.', 1)[0], sh_idx)

            if result_img.shape[0]>min_resolution:
                result_img = resize_pil(result_img, height=min_resolution)

            result_img = np.ascontiguousarray(result_img, dtype=np.uint8)
            cv2.putText(result_img, model_obj.name, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 255)
            results_frame.append(result_img)

        # tgt_result = cv2.resize(tgt_result, (256,256))

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