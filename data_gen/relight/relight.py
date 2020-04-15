#!/usr/bin/python3
# execute in blender!

import sys
import bpy
import os
import math
import shutil
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bprint import print as bprint
from os import listdir
from os.path import join
def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def iterate(root, dst_root):
    image_names = os.listdir(root)

    os.makedirs(dst_root, exist_ok=True)
    countme = 5

    for image_name in image_names:
        print(image_name)
        img_path = join(root, image_name, f"{image_name}.png")
        normal_path = join(root, image_name, f"full_normal_faceRegion_faceBoundary_extend.png")
        dst_dir = join(dst_root, image_name)

        try:

            Relight(img_path, normal_path, dst_dir)

            # validation, in some cases, the images are all black (opengl issue, in such cases you must login to the actual monitor and use export DISPLAY:=0.0)
            first_path = join(dst_dir, "0000.jpg")
            if os.path.getsize(first_path) < 70000:
                raise RuntimeError("ValidationError, Image seems empty!")
        except Exception as e:
            print(f"ERROR executing {image_name}")
            raise e

def Relight(image_path, normal_path, dest_folder):
    image_obj = bpy.data.objects["image"]
    sun_obj = bpy.data.objects["sun"]
    material_relight = bpy.data.materials["relight"]
    nodes = material_relight.node_tree.nodes
    nodes["Image Texture"].image.filepath = image_path
    nodes["Image Texture.001"].image.filepath = normal_path
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    os.makedirs(dest_folder, exist_ok=True)
    with open(join(dest_folder, 'index.txt'), 'w') as f:
        print(f"#name, light_Y, light_Z, light_intensity, world_intensity", file=f)
        for i in range(15):
            sun_obj.rotation_euler[0] = 0 # no point in varying this variable
            sun_obj.rotation_euler[1] = math.pi * .5 + random.uniform(-math.pi / 3, math.pi / 3)  # the default for this axis is 90-degrees
            sun_obj.rotation_euler[2] = random.uniform(-math.pi / 2, math.pi / 2)
            sun_obj.data.energy = 3.9
            sun_obj.data.angle = 0.785398
            bpy.context.scene.world.color = [0.37] * 3
            name = "{:0>4}.jpg".format(i)
            bprint(f"{image_path} : {name}__r")
            print(f"{name},{sun_obj.rotation_euler[1]},{sun_obj.rotation_euler[2]},{sun_obj.data.energy},{bpy.context.scene.world.color[0]}", file=f)
            bpy.context.scene.render.filepath = join(dest_folder, name)
            bpy.ops.render.render(write_still=True, animation=False)
    bprint("") # new line

if __name__ == '__main__':
    argv = sys.argv
    # argv = argv[argv.index("--") + 1:]  # get all args after "--"
    # image_path = argv[0]
    # normal_path = argv[1]
    # dest_folder = argv[2]

    root = "/mnt/data2/users/morris/data_synth"
    #root = "/mnt/data2/users/morris/data_DPR_relight"
    dst_root = "/home/nedko/face_relight/outputs/3dulight_test"
    iterate(root, dst_root)
    

