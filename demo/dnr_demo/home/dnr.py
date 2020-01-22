import glob

from django.conf import settings
from celery import shared_task
from celery.signals import worker_process_init
from billiard import current_process
import sys
import os.path as osp
import os
import dlib
sys.path.append("../..")
sys.path.append("../../model")
import cv2
import imutils
from multiprocessing.pool import ThreadPool

import uuid

from models.skeleton512 import HourglassNet


from .segment_model import BiSeNet
import numpy as np
import torchvision.transforms as transforms

# only load these modules for Celery workers, to not slow down django
if settings.IS_CELERY_WORKER:
    import numpy as np
    import torch

base_model = None
sh_lookups = None
sh_lookup = None
predictor = None
detector = None
segment_model = None
segment_norm = None

def get_device():
    worker_id = current_process().index
    if worker_id < 2:
        device = "cuda:0"
    else:
        device = "cpu"

    # device = "cuda:0"
    return device

def init_gpu(data_path, model_path):
    global base_model, sh_lookup, sh_lookups, detector, predictor, segment_model, segment_norm

    device = get_device()
    if base_model is None:
        base_model = HourglassNet()
        base_model.load_state_dict(torch.load(model_path))
        base_model.to(device)
        base_model.train(False)

        print("Worker {} ready".format(current_process().index))

    if sh_lookups is None:

        sh_dir = osp.join(data_path, 'sh_presets')
        sh_lookups = {}

        for dir in os.listdir(sh_dir):
            sh_instance_dir = osp.join(sh_dir, dir)
            n = len(glob.glob(osp.join(sh_instance_dir,'*.txt')))
            sh_lookups[dir] = []
            for i in range(n):
                sh = np.loadtxt(osp.join(sh_instance_dir, 'rotate_light_{:02d}.txt'.format(i)))
                sh = sh[0:9]
                sh_lookups[dir].append(sh)

        sh_lookup = sh_lookups['horizontal']

    if detector is None:
        detector = dlib.get_frontal_face_detector()

    if predictor is None:
        lmarks_model_path = osp.join(data_path, 'model', 'shape_predictor_68_face_landmarks.dat')
        predictor = dlib.shape_predictor(lmarks_model_path)

    if segment_model is None:
        n_classes = 19
        segment_model = BiSeNet(n_classes=n_classes)
        segment_model.to(device)
        segment_model_path = osp.join(data_path, 'model', 'face_parsing.pth')
        segment_model.load_state_dict(torch.load(segment_model_path))
        segment_model.eval()

    if segment_norm is None:
        segment_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

R_90 = R(np.deg2rad(90))

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def vis_parsing_maps(im, parsing_anno, stride,h=None,w=None):

    im = np.array(im)
    alpha_2 = np.zeros((h,w,3))
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # MASK
    vis_parsing_anno = cv2.resize(vis_parsing_anno,(w,h))
    vis_parsing_anno[vis_parsing_anno==16]=0
    vis_parsing_anno[vis_parsing_anno==14]=0
    vis_parsing_anno[vis_parsing_anno>0]=255
    vis_parsing_anno = cv2.GaussianBlur(vis_parsing_anno,(9,9),15,15)

    th, im_th = cv2.threshold(vis_parsing_anno, 244.5, 255, cv2.THRESH_BINARY_INV)
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_floodfill = im_th.copy()
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    cv2.floodFill(im_floodfill, mask, (w-1, 0), 255)
    cv2.floodFill(im_floodfill, mask, (0, h-1), 255)
    cv2.floodFill(im_floodfill, mask, (w-1, h-1), 255)

    mask = mask[0:0+h, 0:0+w]
    vis_parsing_anno = cv2.bitwise_not(mask)
    vis_parsing_anno[vis_parsing_anno==254]=0

    alpha_2[:,:,0] = np.copy(vis_parsing_anno)
    alpha_2[:,:,1] = np.copy(vis_parsing_anno)
    alpha_2[:,:,2] = np.copy(vis_parsing_anno)

    kernel = np.ones((10, 10), np.uint8)
    alpha_2 = cv2.erode(alpha_2, kernel, iterations=1)
    alpha_2 = cv2.GaussianBlur(alpha_2,(29,29),15,15)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha_2 = alpha_2.astype(float) / 255
    return alpha_2


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

def handleOutput(outputImg, Lab, col, row, filepath, mask, img_p, img_orig, loc, crop_sz, border):

    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)  # *1.45
    outputImg = (outputImg * 255.0).astype(np.uint8)

    t_Lab = Lab.copy()
    t_Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(t_Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # do something here
    # make a gauss blur

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TODO add original image which will be background  CHECK AGAIN  !!!!!!!!!!!

    background = np.copy(img_p)
    foreground = np.copy(resultLab)
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(mask, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1 - mask, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)


    outImage = cv2.resize(outImage, (crop_sz[1], crop_sz[0]))

    top, bottom, left, right = border
    outImage = outImage[top:-bottom,left:-right]

    img_orig[loc[0]:loc[0]+outImage.shape[0], loc[1]:loc[1]+outImage.shape[1]] = outImage

    cv2.imwrite(filepath, img_orig)
    return True


def preprocess(img, device):
    img = np.array(img)
    orig_size = img.shape

    if np.max(img.shape[:2]) > 1024:
        if img.shape[0] < img.shape[1]:
            img_res = imutils.resize(img, width=1024)
        else:
            img_res = imutils.resize(img, width=1024)
    else:
        img_res = img

    resize_ratio = orig_size[0]/img_res.shape[0]
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    rects, scores, idx = detector.run(gray, 1, 1)

    loc = [0,0]

    if len(rects) > 0:
        mask = segment(np.array(img), device)

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
        mask = mask[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        border = [top, bottom, left, right]

        crop_sz = img.shape
        if np.max(img.shape[:2]) > 1024:
            img = cv2.resize(img, (1024,1024))
            mask = cv2.resize(mask, (1024,1024))
    else:
        img = None
        mask = None
        crop_sz = None
        border = None

    return img, mask, loc, crop_sz, border

@shared_task
def prediction_task_sh_mul(data_path, img_path, preset_name, sh_id, upload_id=None):
    global sh_lookup, base_model
    worker_device = get_device()

    dir_uuid = upload_id
    if dir_uuid == None:
        dir_uuid = str(uuid.uuid1())

    out_dir = osp.join(data_path, 'output', dir_uuid)
    os.makedirs(out_dir, exist_ok=True)

    img_orig = cv2.imread(img_path)

    img_p, mask, loc, crop_sz, border = preprocess(img_orig, worker_device)
    is_face_found = img_p is not None

    if not is_face_found:
        print('FACE NOTE FOUND! Input image path:', img_path)
    else:
        pool = ThreadPool(processes=8)
        pool.apply_async(
            cv2.imwrite,
            [osp.join(out_dir, 'ori.jpg'), img_orig]
        )

        row, col, _ = img_p.shape
        img = cv2.resize(img_p, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]

        for sh_iter in range(50, 105, 10):
            sh_mul = sh_iter * .01
            sh = np.squeeze(sh_lookups[preset_name][sh_id])
            sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32) * sh_mul
            sh = torch.autograd.Variable(torch.from_numpy(sh).to(worker_device))

            t_inputL = torch.autograd.Variable(torch.from_numpy(inputL).to(worker_device))
            outputImg, _, outputSH, _ = base_model(t_inputL, sh, 0)

            outputImg = outputImg[0].cpu().data.numpy()
            filename = preset_name + '_' + str(sh_id) + '_' + ("%.2f" % sh_mul) + '.jpg'
            filepath = osp.join(out_dir, filename)

            pool.apply_async(
                handleOutput,
                [outputImg, Lab, col, row, filepath,mask,img_p, img_orig, loc, crop_sz, border]
            )

        pool.close()
        pool.join()

    return [dir_uuid, is_face_found]

@shared_task
def prediction_task(data_path, img_path, sh_mul=None, upload_id=None):
    global sh_lookup, base_model
    worker_device = get_device()

    dir_uuid = upload_id
    if dir_uuid == None:
        dir_uuid = str(uuid.uuid1())

    out_dir = osp.join(data_path, 'output', dir_uuid)
    os.makedirs(out_dir, exist_ok=True)

    if sh_mul == None:
        sh_mul = 0.7

    img_orig = cv2.imread(img_path)

    img_p, mask, loc, crop_sz, border = preprocess(img_orig, worker_device)
    is_face_found = img_p is not None

    if not is_face_found:
        print('FACE NOTE FOUND! Input image path:', img_path)
    else:

        pool = ThreadPool(processes=8)
        pool.apply_async(
            cv2.imwrite,
            [osp.join(out_dir, 'ori.jpg'), img_orig]
        )

        row, col, _ = img_p.shape
        img = cv2.resize(img_p, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]

        for preset_id, sh_presets in sh_lookups.items():
            for i, sh in enumerate(sh_presets):
                filename = preset_id + '_' + str(i) + '_' + ("%.2f" % sh_mul) + '.jpg'
                filepath = osp.join(out_dir, filename)

                if os.path.exists(filepath):
                    continue

                sh = np.squeeze(sh) * sh_mul
                sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
                sh = torch.autograd.Variable(torch.from_numpy(sh).to(worker_device))

                t_inputL = torch.autograd.Variable(torch.from_numpy(inputL).to(worker_device))

                outputImg, _, outputSH, _ = base_model(t_inputL, sh, 0)
                outputImg = outputImg[0].cpu().data.numpy()

                pool.apply_async(
                        handleOutput,
                        [outputImg, Lab, col, row, filepath, mask, img_p, img_orig, loc, crop_sz, border]
                    )
                # cv2.imwrite(osp.join(out_dir, filename), resultLab)

        pool.close()
        pool.join()

    return [dir_uuid, is_face_found]


# replace this function with your own
# returns the classification result of a given image_path
def process_image(img_path, sh_mul=None, upload_id=None):

    data_path = osp.abspath('../data/')
    task = prediction_task.delay(data_path, img_path, sh_mul, upload_id)

    return task.get()

def process_image_sh_mul(img_path, preset_name, sh_id, upload_id=None):

    data_path = osp.abspath('../data/')
    task = prediction_task_sh_mul.delay(data_path, img_path, preset_name, sh_id, upload_id)

    return task.get()


########## INIT ###########

@worker_process_init.connect
def worker_process_init_(**kwargs):

    data_path = osp.abspath('../data/')
    model_path = osp.join(data_path, "model/14_net_G_dpr7_mseBS20.pth")
    init_gpu(data_path, model_path)  # make sure all models are initialized upon starting the worker
    # prediction_task_sh_mul(data_path, '../../test_data/portrait_/AJ.jpg', 'horizontal', 7)
    # prediction_task(data_path, '../../test_data/portrait_/mal.jpg')
    # prediction_task(data_path, '../../test_data/01/rotate_light_00.png')