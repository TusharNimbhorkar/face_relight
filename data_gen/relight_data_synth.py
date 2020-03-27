#!/usr/bin/python
# relight deepserver2@/share/datasets/data_synth
import os
from relight import relight
from os.path import join

root = "/mnt/data2/users/morris/data_synth"
dst_root = "/mnt/data2/users/morris/data_synth_relight2"
image_names = os.listdir(root)

os.makedirs(dst_root, exist_ok=True)

for image_name in image_names:
    img_path = join(root, image_name, f"{image_name}.png")
    normal_path = join(root, image_name, f"full_normal_faceRegion_faceBoundary_extend.png")
    dst_dir = join(dst_root, image_name)

    try:
        relight(img_path, normal_path, dst_dir, "datagen.blend", False)
        
        # validation, in some cases, the images are all black (opengl issue, in such cases you must login to the actual monitor and use export DISPLAY:=0.0)
        first_path = join(dst_dir, "0000.jpg")
        if os.path.getsize(first_path) < 70000:
            raise RuntimeError("ValidationError, Image seems empty!")        
    except:
        print(f"ERROR executing {image_name}")
