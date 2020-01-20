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
import torch

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
        # # --------------------------------------------------
        # # rendering half-sphere
        # sh = np.squeeze(sh)
        # shading = get_shading(normal, sh)
        # value = np.percentile(shading, 95)
        # ind = shading > value
        # shading[ind] = value
        # shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        # shading = (shading * 255.0).astype(np.uint8)
        # shading = np.reshape(shading, (256, 256))
        # shading = shading * valid
        #
        # sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
        # sh = Variable(torch.from_numpy(sh).cuda())
        #
        # millis_start = int(round(time.time() * 1000))
        #
        # outputImg, _, outputSH, _ = my_network(inputL, sh, skip_c)
        #
        # # outputImg, _, outputSH, _ = my_network(inputL, outputSH, skip_c)
        #
        # millis_after = int(round(time.time() * 1000))
        # elapsed = millis_after - millis_start
        # print('MILISECONDS:  ', elapsed)
        # time_avg += elapsed
        # count = count + 1
        # outputImg = outputImg[0].cpu().data.numpy()
        # outputImg = outputImg.transpose((1, 2, 0))
        # outputImg = np.squeeze(outputImg)  # *1.45
        # outputImg = (outputImg * 255.0).astype(np.uint8)
        # Lab[:, :, 0] = outputImg
        # resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        # resultLab = cv2.resize(resultLab, (col, row))
        # cv2.imwrite(os.path.join(saveFolder, im[:-4] + '_{:02d}.jpg'.format(i)), resultLab)

@shared_task
def prediction_task(data_path, x,y):
    worker_device = get_device()

    sh = sh_lookup[x]

    filename_uuid = str(uuid.uuid1())
    print("Out size:", base_model.out_size)
    _, pred = base_model()

    pred = (1+pred.cpu().detach())/2
    pred = np.transpose(pred.numpy()[0]*255, (1,2,0))
    # pred = np.zeros((512,512,3))

    filename = filename_uuid +".png"
    cv2.imwrite(osp.join(data_path,'out/'+filename), pred[:,:,::-1])

    return filename


# replace this function with your own
# returns the classification result of a given image_path
def process_image(x_id, y_id=0):

    # global data_path, src_data_path, model_path
    #
    data_path = osp.abspath('../data/')
    task = prediction_task.delay(data_path, x_id, y_id)

    return task.get()


########## INIT ###########

@worker_process_init.connect
def worker_process_init_(**kwargs):

    data_path = osp.abspath('../data/')
    model_path = osp.join(data_path, "model/14_net_G_dpr7_mseBS20.pth")
    init_gpu(data_path, model_path)  # make sure all models are initialized upon starting the worker
