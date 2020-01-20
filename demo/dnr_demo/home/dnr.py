import glob

from django.conf import settings
from celery import shared_task
from celery.signals import worker_process_init
from billiard import current_process
import sys
import os.path as osp
import os
sys.path.append("../..")
sys.path.append("../../model")
import cv2

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

def get_device():
    worker_id = current_process().index
    if worker_id < 2:
        device = "cuda:0"
    else:
        device = "cpu"

    return device

def init_gpu(data_path, model_path):
    global base_model, sh_lookup, sh_lookups
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
            for i in range(n-1):
                sh = np.loadtxt(osp.join(sh_instance_dir, 'rotate_light_{:02d}.txt'.format(i)))
                sh = sh[0:9]
                sh_lookups[dir].append(sh)

        sh_lookup = sh_lookups['horizontal']

@shared_task
def prediction_task(data_path, img_path):
    global sh_lookup, base_model
    worker_device = get_device()

    dir_uuid = str(uuid.uuid1())
    out_dir = osp.join(data_path, 'output', dir_uuid)
    os.makedirs(out_dir, exist_ok=True)

    for preset_id, sh_presets in sh_lookups.items():
        for i, sh in enumerate(sh_presets):
            sh_mul = 0.8
            sh = np.squeeze(sh) * sh_mul
            sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
            sh = torch.autograd.Variable(torch.from_numpy(sh).to(worker_device))

            img = cv2.imread(img_path)
            row, col, _ = img.shape
            img = cv2.resize(img, (512, 512))
            Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            inputL = Lab[:, :, 0]
            inputL = inputL.astype(np.float32) / 255.0
            inputL = inputL.transpose((0, 1))
            inputL = inputL[None, None, ...]
            inputL = torch.autograd.Variable(torch.from_numpy(inputL).to(worker_device))

            outputImg, _, outputSH, _ = base_model(inputL, sh, 0)

            outputImg = outputImg[0].cpu().data.numpy()
            outputImg = outputImg.transpose((1, 2, 0))
            outputImg = np.squeeze(outputImg)  # *1.45
            outputImg = (outputImg * 255.0).astype(np.uint8)
            Lab[:, :, 0] = outputImg
            resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
            resultLab = cv2.resize(resultLab, (col, row))

            filename = preset_id + '_' + str(i) + '.png'
            cv2.imwrite(osp.join(out_dir, filename), resultLab)

    return dir_uuid


# replace this function with your own
# returns the classification result of a given image_path
def process_image(img_path):

    # global data_path, src_data_path, model_path
    #
    data_path = osp.abspath('../data/')
    task = prediction_task.delay(data_path, img_path)

    return task.get()


########## INIT ###########

@worker_process_init.connect
def worker_process_init_(**kwargs):

    data_path = osp.abspath('../data/')
    model_path = osp.join(data_path, "model/14_net_G_dpr7_mseBS20.pth")
    init_gpu(data_path, model_path)  # make sure all models are initialized upon starting the worker
    # prediction_task(data_path, '../../test_data/portrait_/a1.jpeg', 0,0)
