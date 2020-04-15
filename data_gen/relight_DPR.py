#!/usr/bin/python
# relight deepserver2@/share/datasets/data_synth
# export DISPLAY=:0.0
import os
from relight import relight
from os.path import join, isfile

root = "/mnt/data2/users/morris/data_DPR_relight"
dst_root = "/home/nedko/face_relight/outputs/3dulight_test"
image_names = os.listdir(root)

os.makedirs(dst_root, exist_ok=True)
countme = 5

for image_name in image_names:
    print(image_name)
    img_path = join(root, image_name, f"frame.jpg")
    normal_path = join(root, image_name, f"normals_warped.png")
    if isfile(normal_path):
        dst_dir = join(dst_root, image_name)

        relight(img_path, normal_path, dst_dir, True)
        
        # validation, in some cases, the images are all black (opengl issue, in such cases you must login to the actual monitor and use export DISPLAY:=0.0)
        first_path = join(dst_dir, "0000.jpg")
        if os.path.getsize(first_path) < 70000:
            raise RuntimeError("ValidationError, Image seems empty!")
        countme -= 1
        if countme < 0:
            break;
    else:
        print(f"could not process {normal_path}")
    
