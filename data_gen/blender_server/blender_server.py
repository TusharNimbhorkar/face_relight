# ==========================================
#  Project:  Blender Common Tools
#  Author: Nedko Savov
#  Date:
# ==========================================

from .tcp_server import TCPServer
from .config import *

import glob
import os

import bpy
import numpy as np
import math
import time
import sys

sys.path.append('/home/nedko/data2/tools/deferred_rendering/model')



def init():
    global bounding_box_path, base_path, tree
    n_args = len(sys.argv) - sys.argv.index('--') - 1

    if n_args == 2 and sys.argv[-1] == '-demo_setup':
        is_demo_setup = True
        base_path = sys.argv[-2]  # '/home/nedko/data2/tools/deferred_rendering/data/nike_1_2'
    else:
        is_demo_setup = False
        base_path = sys.argv[-1]

    model_path = base_path + '/mesh_cleaned'
    bounding_box_path = base_path + '/bounding_box_rot.txt'
    dest_uv_dir = base_path + '/uvs'
    uv_map_dir = os.path.join(dest_uv_dir, "exr")
    dest_norms_dir = base_path + '/norms'
    norms_map_dir = os.path.join(dest_norms_dir, "exr")


    # Find the model
    model_path = glob.glob(model_path + '.*')[0]

    tree = bpy.context.scene.node_tree

    tree.nodes["Norm_out"].base_path = norms_map_dir
    tree.nodes["UV_out"].base_path = uv_map_dir


    blender_scene_utils.load_object(model_path)
    obj = bpy.data.objects[0]

    if is_demo_setup:
        bpy.context.scene.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        obj.location = [0, 0, 0]

    # create folder to store the uvs
    if not os.path.exists(uv_map_dir):
        os.makedirs(uv_map_dir)
    if not os.path.exists(norms_map_dir):
        os.makedirs(norms_map_dir)


    intrinsics_path = base_path + '/intrinsics.txt'
    intrinsic = read_intrinsic_parameters(intrinsics_path)
    bpy.context.scene.render.resolution_x = intrinsic['width']
    bpy.context.scene.render.resolution_y = intrinsic['height']

    cam = blender_scene_utils.create_camera(intrinsic)
    bpy.context.scene.camera = cam

def render(ip, queue, data):
    out_uuid = data.decode("utf-8")
    extrinsics_path = base_path + '/pose/' + out_uuid + '.txt'
    is_from_metashape = False if 'synthetic' in base_path else True
    is_from_metashape = False
    is_demo_setup = True
    cam = bpy.context.scene.camera

    if is_demo_setup:
        tree.nodes["Norm_out"].file_slots[0].path = "Norms_" + out_uuid + '_'
        tree.nodes["UV_out"].file_slots[0].path = "UVs_" + out_uuid + '_'

    extrinsic = [read_matrix(extrinsics_path)]
    # create only 1 camera and adjusst its position at each iteration

    if is_demo_setup:
        if os.path.exists(bounding_box_path):
            T_align = read_matrix(bounding_box_path)
            T_align[:, 0] /= np.linalg.norm(T_align[:, 0])
            T_align[:, 1] /= np.linalg.norm(T_align[:, 1])
            T_align[:, 2] /= np.linalg.norm(T_align[:, 2])
            # T_align[0,0] = 1
            # T_align[1, 1] = 1
            # T_align[2, 2] = 1
            # T_align = np.fill_diagonal(T_align, 1)
            T_align = T_align.transpose()
        else:
            T_align = np.eye(4)
            # cam.matrix_world = cam.linalg.inv(T_align)


    for view_id, ext in enumerate(extrinsic):

        start = time.time()

        if is_demo_setup:
            cam.matrix_world = np.dot(ext.transpose(),
                                      T_align)
        else:
            cam.matrix_world = ext.transpose()



        # fix orientation of cameras.
        # Probably due to different conventions between
        # Blender and Metashape.
        if is_from_metashape and not is_demo_setup:
            cam.rotation_euler[0] += math.pi
            bpy.context.scene.update()

        bpy.context.scene.frame_set(view_id)

        ##Silent render
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)
        bpy.ops.render.render()
        os.close(1)
        os.dup(old)
        os.close(old)

        end = time.time()
        print("View {:3d}/{} rendered in {:5.3f} sec".format(view_id, len(extrinsic), end - start))
        queue.put(b"1")


if __name__ == "__main__":
    init()

    server = TCPServer(IP_ADDRESS, PORT, render)
    server.run()