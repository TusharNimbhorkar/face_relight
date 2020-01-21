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
from multiprocessing.pool import ThreadPool

import uuid

from models.skeleton512 import HourglassNet
import numpy as np

# only load these modules for Celery workers, to not slow down django
if settings.IS_CELERY_WORKER:
    import numpy as np
    import torch

base_model = None
sh_lookups = None
sh_lookup = None
predictor = None
detector = None

def get_device():
    worker_id = current_process().index
    if worker_id < 2:
        device = "cuda:0"
    else:
        device = "cpu"

    return device

def init_gpu(data_path, model_path):
    global base_model, sh_lookup, sh_lookups, detector, predictor
    if base_model is None:
        device = get_device()

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

def handleOutput(outputImg, Lab, col, row, filepath):
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)  # *1.45
    outputImg = (outputImg * 255.0).astype(np.uint8)

    t_Lab = Lab.copy()
    t_Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(t_Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, resultLab)
    return True


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects, scores, idx = detector.run(gray, 1, 1)

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

        c1_ms = np.max([0, int(c[1] - s / 2)])
        c1_ps = np.max([0, int(c[1] + s / 2)])
        c0_ms = np.max([0, int(c[0] - s / 2)])
        c0_ps = np.max([0, int(c[0] + s / 2)])

        img = img[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
    else:
        img = None

    return img

@shared_task
def prediction_task_sh_mul(data_path, img_path, preset_name, sh_id):
    global sh_lookup, base_model
    worker_device = get_device()

    dir_uuid = str(uuid.uuid1())
    out_dir = osp.join(data_path, 'output', dir_uuid)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)

    img = preprocess(img)
    is_face_found = img is not None

    if not is_face_found:
        print('FACE NOTE FOUND! Input image path:', img_path)
    else:
        pool = ThreadPool(processes=8)
        pool.apply_async(
            cv2.imwrite,
            [osp.join(out_dir, 'ori.jpg'), img]
        )

        row, col, _ = img.shape
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]

        for sh_mul in np.arange(0.5, 1.6, 0.1):
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
                [outputImg, Lab, col, row, filepath]
            )

        pool.close()
        pool.join()

    return [dir_uuid, is_face_found]

@shared_task
def prediction_task(data_path, img_path, sh_mul=None):
    global sh_lookup, base_model
    worker_device = get_device()

    dir_uuid = str(uuid.uuid1())
    out_dir = osp.join(data_path, 'output', dir_uuid)
    os.makedirs(out_dir, exist_ok=True)
    is_face_found = True

    if sh_mul == None:
        sh_mul = 0.7

    img = cv2.imread(img_path)

    img = preprocess(img)
    is_face_found = img is not None

    if not is_face_found:
        print('FACE NOTE FOUND! Input image path:', img_path)
    else:

        pool = ThreadPool(processes=8)
        pool.apply_async(
            cv2.imwrite,
            [osp.join(out_dir, 'ori.jpg'), img]
        )

        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]

        for preset_id, sh_presets in sh_lookups.items():
            for i, sh in enumerate(sh_presets):
                sh = np.squeeze(sh) * sh_mul
                sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
                sh = torch.autograd.Variable(torch.from_numpy(sh).to(worker_device))

                t_inputL = torch.autograd.Variable(torch.from_numpy(inputL).to(worker_device))

                outputImg, _, outputSH, _ = base_model(t_inputL, sh, 0)

                outputImg = outputImg[0].cpu().data.numpy()
                filename = preset_id + '_' + str(i) + '_' + str(sh_mul) + '.jpg'
                filepath = osp.join(out_dir, filename)

                pool.apply_async(
                        handleOutput,
                        [outputImg, Lab, col, row, filepath]
                    )
                # cv2.imwrite(osp.join(out_dir, filename), resultLab)

        pool.close()
        pool.join()

    return [dir_uuid, is_face_found]


# replace this function with your own
# returns the classification result of a given image_path
def process_image(img_path):

    data_path = osp.abspath('../data/')
    task = prediction_task.delay(data_path, img_path)

    return task.get()

def process_image_sh_mul(img_path, preset_name, sh_id):

    data_path = osp.abspath('../data/')
    task = prediction_task_sh_mul.delay(data_path, img_path, preset_name, sh_id)

    return task.get()


########## INIT ###########

@worker_process_init.connect
def worker_process_init_(**kwargs):

    data_path = osp.abspath('../data/')
    model_path = osp.join(data_path, "model/14_net_G_dpr7_mseBS20.pth")
    init_gpu(data_path, model_path)  # make sure all models are initialized upon starting the worker
    # prediction_task_sh_mul(data_path, '../../test_data/portrait_/AJ.jpg', 'horizontal', 7)
    prediction_task(data_path, '../../test_data/portrait_/AJ.jpg')
    # prediction_task(data_path, '../../test_data/01/rotate_light_00.png')



#
#
# #!/usr/bin/python
# # -*- encoding: utf-8 -*-
#
# from model import BiSeNet
#
# import torch
#
# import os
# import os.path as osp
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import cv2
#
# def vis_parsing_maps(im, parsing_anno, stride,h=None,w=None):
#
#     im = np.array(im)
#     vis_im = im.copy().astype(np.uint8)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
#     num_of_class = np.max(vis_parsing_anno)
#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
#     # MASK
#     vis_parsing_anno = cv2.resize(vis_parsing_anno,(w,h))
#     vis_parsing_anno[vis_parsing_anno==16]=0
#     vis_parsing_anno[vis_parsing_anno==16]=0
#     vis_parsing_anno[vis_parsing_anno>0]=1
#     return vis_parsing_anno
#
#
# def evaluate(img):
#
#     n_classes = 19
#     net = BiSeNet(n_classes=n_classes)
#     net.cuda()
#     save_pth = osp.join('cp', '79999_iter.pth')
#     net.load_state_dict(torch.load(save_pth))
#     net.eval()
#
#     to_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     with torch.no_grad():
#
#         w, h = img.size
#         image = img.resize((512, 512), Image.BILINEAR)
#         img = to_tensor(image)
#         img = torch.unsqueeze(img, 0)
#         img = img.cuda()
#         out = net(img)[0]
#         parsing = out.squeeze(0).cpu().numpy().argmax(0)
#         output_img = vis_parsing_maps(image, parsing, stride=1,h=h,w=w)
#     return output_img
#
#
# if __name__ == "__main__":
#     image_path = 'data/aa.jpg'
#     img = Image.open(osp.join(image_path))
#     segment = evaluate(image=img)

