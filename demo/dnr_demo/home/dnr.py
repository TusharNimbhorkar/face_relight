from django.conf import settings
from celery import shared_task
from celery.signals import worker_process_init
from billiard import current_process
import sys
import os.path as osp
sys.path.append("../..")
sys.path.append("../../model")
from model.utils import preprocess, load_model
import cv2
from scipy.spatial.transform import Rotation
import subprocess
import shutil
import glob
from scene_tools.blender_server.client_socket import ClientSocket
from scene_tools.blender_server.config import *

import uuid 
import numpy as np
import torch

# only load these modules for Celery workers, to not slow down django
if settings.IS_CELERY_WORKER:
    import numpy as np
    import torch
    # from tensorflow.keras.preprocessing import image
    # from tensorflow.keras.applications.inception_v3 import preprocess_input
    # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


image_shape = None  # will be initialised in init_gpu()
imagenet_labels = None
base_model = None


def get_rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])

def get_rot_x(angle):
        return np.array([[1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]])

def get_rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle),0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])


def load_data(data_path, out_uuid, theta, phi, r, out_size):
    pose_dir = osp.join(data_path, 'pose')
    uv_dir = osp.join(data_path, 'uvs/exr')
    norm_dir = osp.join(data_path, 'norms/exr')

    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r*np.sin(theta)*np.cos(phi)
    z = r*np.sin(theta)*np.sin(phi)
    y = r*np.cos(theta)
    
    view_v = np.asarray([0,0,1])
    target_v = np.asarray([x,y,z])

    target_n_v = target_v/np.linalg.norm(target_v)

    if np.count_nonzero(target_n_v-view_v) == 0:
        R = np.eye(3)
    else:

        a=phi
        b=theta
        c=np.deg2rad(-90)

        rot_x = get_rot_x(theta)
        rot_x_90 = get_rot_x(np.deg2rad(-90))

        rot_x2 = np.array([[1, 0, 0],
                [0, np.cos(b), -np.sin(b)],
                [0, np.sin(b), np.cos(b)]])

        rot_y = get_rot_y(-phi)
        rot_y_90 = get_rot_y(np.deg2rad(90))

        rot_z = get_rot_z(-theta)

        R = np.dot(rot_y, np.dot(rot_z, np.dot(rot_y_90, rot_x_90)))
        # cross_v = np.cross(view_v, target_n_v)
        # s = np.linalg.norm(cross_v)
        # c = np.dot(view_v,target_n_v)
        # skew_sym = np.asarray([[0, -cross_v[2], cross_v[1]],[cross_v[2], 0, -cross_v[0]],[-cross_v[1], cross_v[0], 0]])
        # R = np.eye(3) + skew_sym + np.dot(skew_sym, skew_sym)*(1-c)/(s**2)
        # print(R, s)

    # x,y,z = np.dot(R, target_v)

    T = np.eye(4)
    T[0,-1] = x
    T[1,-1] = y
    T[2,-1] = z
    T[:3, :3] = R

    
    np.savetxt(osp.join(pose_dir, "%s.txt" % out_uuid), np.reshape(T, (1,-1)))

    out_prefix = "%s" % str(out_uuid)
    # subprocess.call(["bash", osp.join(base_path, "blender_gen_maps.sh"), data_path, "-demo_setup", out_prefix])

    blender_socket = ClientSocket(IP_ADDRESS, PORT)
    blender_socket.send(out_prefix)

    out_filename = "%s_0000.exr" % str(out_uuid)

    uv_filename = "UVs_" + out_filename
    uv_paths = [osp.join(uv_dir, uv_filename)]

    norms_filename = "Norms_" + out_filename
    norm_paths = [osp.join(norm_dir, norms_filename)]

    _, uv, norm = preprocess(uv_paths, norm_paths, end_height=out_size[1], end_width=out_size[0])
    uv = torch.from_numpy(uv[0]).unsqueeze(0)
    norm = torch.from_numpy(norm[0]).unsqueeze(0)
    
    return uv, norm

def get_device():
    worker_id = current_process().index
    if worker_id < 2:
        device = "cuda:0"
    else:
        device = "cpu"

    return device

def init_gpu(data_path, src_data_path, model_path):
    global base_model, image_shape, base_path

    if base_model is None:
        device = get_device()
        base_path = osp.abspath("../../")
        mesh_path = osp.join(src_data_path, "mesh_cleaned.ply")
        bb_path = osp.join(src_data_path, "bounding_box_rot.txt")
        intrinsics_path = osp.join(src_data_path, "intrinsics.txt")

        shutil.copy(mesh_path, data_path)
        shutil.copy(bb_path, data_path)
        shutil.copy(intrinsics_path, data_path)

        base_model = load_model(model_path, device).to(device)

        image_shape = [base_model.out_size[1],base_model.out_size[0],3]
        print("Worker {} ready".format(current_process().index))


@shared_task
def prediction_task(data_path, x,y,z):
    worker_device = get_device()

    filename_uuid = str(uuid.uuid1())
    print("Out size:", base_model.out_size)
    uv, norm = load_data(data_path, filename_uuid, x,y,z, base_model.out_size)
    _, pred = base_model(uv.to(worker_device), norm.to(worker_device))

    pred = (1+pred.cpu().detach())/2
    pred = np.transpose(pred.numpy()[0]*255, (1,2,0))
    # pred = np.zeros((512,512,3))

    filename = filename_uuid +".png"
    cv2.imwrite(osp.join(data_path,'out/'+filename), pred[:,:,::-1])

    return filename


# replace this function with your own
# returns the classification result of a given image_path
def process_image(x,y,z):

    # global data_path, src_data_path, model_path
    #
    data_path = osp.abspath('../data/')
    # src_data_path = "/home/nedko/deferred_rendering/data/nike_shoe_camera/nike_shoe_clean/out/model_2048_v2_lr5e-3_scale/"
    # model_path = osp.join(src_data_path,"out/model_2048_v2_lr5e-3_scale/model_data__epoch_10.pt")
    # init_gpu(data_path, src_data_path, model_path)
    # pred1 = prediction_task(data_path,x,y,z)

    task = prediction_task.delay(data_path, x,y,z)

    # base_model = None
    # data_path2 = osp.abspath('../data2/')
    # src_data_path2 = "/home/nedko/data2/tools/deferred_rendering/data/finde_shoe_pixel/finde_shoe_pixel/"
    # model_path2 = osp.join(src_data_path2,"out/model_2048_cont/model_data__epoch_27.pt")
    # init_gpu(data_path2, src_data_path2, model_path2)
    # pred2 = prediction_task(data_path2,x,y,z)

    # pred1 = np.ones((512,512,3))
    # cat_pred = pred1 #np.concatenate((pred1, pred2), axis=1)
    # filename = str(uuid.uuid1())+".png"
    # cv2.imwrite(osp.join(data_path,'out/'+filename), cat_pred[:,:,::-1])


    return task.get()


########## INIT ###########

@worker_process_init.connect
def worker_process_init_(**kwargs):

    data_path = osp.abspath('../data/')
    src_data_path = "/home/nedko/data2/tools/deferred_rendering/data/nike_shoe_camera/nike_shoe_clean/"
    model_path = osp.join(src_data_path, "out/model_2048_v2_o1024/model_data__epoch_6.pt")
    init_gpu(data_path, src_data_path, model_path)  # make sure all models are initialized upon starting the worker
