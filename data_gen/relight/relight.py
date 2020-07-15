#!/usr/bin/python3
# execute in blender!
# requires natsort (--target=/home/morris/.config/blender/2.82/scripts/addons2/modules)

import sys
import bpy
import os
import math
import shutil
import random
import traceback
from natsort import natsorted
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bprint import print as bprint
from os import listdir
from os.path import join
import os
import math
from gen_ambient_illumination import get_ambient_color, get_sun_color

def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def MultiRelight(root, dst_root, frame_name, normals_name, start_idx, params):
    image_names = natsorted(os.listdir(root))
    # image_names = image_names[28498:]
    os.makedirs(dst_root, exist_ok=True)
    
    bpy.context.scene.render.resolution_x = params['size']
    bpy.context.scene.render.resolution_y = params['size']
    # ~ bpy.context.scene.render.image_settings.quality = 98

    count = 0
    nr_renders_per_image = 5
    for idx, image_name in enumerate(image_names[start_idx:]):
        if normals_name == "full_normal_faceRegion_faceBoundary_extend.png" or normals_name == "full_normal.png": # For DPR / stylegan
            img_path = join(root, image_name, f"{image_name}.png")
        else:
            img_path = join(root, image_name, frame_name)
        normal_path = join(root, image_name, normals_name)
        dst_dir = join(dst_root, image_name)
        first_path = join(dst_dir, "0000.jpg")
        last_path = join(dst_dir, "{:0>4}.jpg".format(nr_renders_per_image-1))

        # skip already processed images:
        if os.path.isfile(last_path):
            bprint(f"skipping {img_path}")
            continue

        try:
            Relight(img_path, normal_path, dst_dir, params, nr_renders_per_image)
            
            # validation, in some cases, the images are all black (opengl issue, in such cases you must login to the actual monitor and use export DISPLAY:=0.0)
            if os.path.getsize(first_path) < 2000:# 70000
                raise RuntimeError("ValidationError, Image seems empty!")                        
            count += 1
            if count > 100:
                bprint(f"{start_idx + idx + 1}__break")
                return
        except:
            bprint(f"ERROR executing {img_path}")
            bprint(traceback.format_exc())


def Relight(image_path, normal_path, dest_folder, params, nr_renders_per_image):
    image_obj = bpy.data.objects["image"]
    sun_obj = bpy.data.objects["sun"]
    material_relight = bpy.data.materials["relight"]
    nodes = material_relight.node_tree.nodes
    bpy.data.images.remove(nodes["Image Texture"].image)
    bpy.data.images.remove(nodes["Image Texture.001"].image)    
    img_color = bpy.data.images.load(image_path)  
    img_normal = bpy.data.images.load(normal_path)
    img_normal.colorspace_settings.name = 'Non-Color'
    nodes["Image Texture"].image = img_color
    nodes["Image Texture.001"].image = img_normal
            
    os.makedirs(dest_folder, exist_ok=True)
    with open(join(dest_folder, 'index.txt'), 'w') as f:
        print(f"#name, light_Y, light_Z, light_intensity, ambient_r,ambient_g,ambient_b, sun_angle, sun_r, sun_g, sun_b", file=f)
        for i in range(nr_renders_per_image):
            # color = get_sun_color()
            ambient_strength = random.uniform(0.04, 0.4) #0.28
            
            sun_obj.rotation_euler[0] = 0 # no point in varying this variable
            sun_obj.rotation_euler[1] = math.pi * .5 + (random.uniform(-math.pi * 60 / 180, math.pi * 45 / 180))  # the default for this axis is 90-degrees 
            sun_obj.rotation_euler[2] = random.uniform(-math.pi / 2, math.pi / 2)
            sun_obj.data.energy = random.uniform(4,7)  # [4,7] # params['intensity']
            sun_obj.data.angle = 60 * math.pi / 180 # params['sun_angle'] # 45 degrees, default is 11.4 degrees (0.198968 radian)
            # sun_obj.data.color = color
            
            bpy.context.scene.world.color = [ambient_strength,] * 3
            name = "{:0>4}.jpg".format(i)
            bprint(f"{image_path} : {name}__r")
            print(f"{name},{sun_obj.rotation_euler[1]},{sun_obj.rotation_euler[2]},{sun_obj.data.energy},{bpy.context.scene.world.color[0]},{bpy.context.scene.world.color[1]},{bpy.context.scene.world.color[2]},{sun_obj.data.angle},{sun_obj.data.color[0]},{sun_obj.data.color[1]},{sun_obj.data.color[2]}", file=f)
            bpy.context.scene.render.filepath = join(dest_folder, name)
            bpy.ops.render.render(write_still=True, animation=False)
    bprint("") # new line

if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    root = argv[0]
    dst_root = argv[1]
    frame_name = argv[2]
    normals_name = argv[3]
    start_idx = 0 if len(argv) <= 4 else int(argv[4])
    # ~ bprint(argv)
    params = {        
    'size' : 256 if len(argv) <= 5 else int(argv[5]),
    'world_color' : 0.17 if len(argv) <= 6 else float(argv[6]),
    'intensity' : 4.2 if len(argv) <= 7 else float(argv[7]),
    'sun_angle' : 0.785398 if len(argv) <= 8 else float(argv[8]), # 35: 0.6108652381980153, 60: 1.0471975511965976
    }
    bprint(params)
    MultiRelight(root, dst_root, frame_name, normals_name, start_idx, params)
    

