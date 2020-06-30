#!/usr/bin/python
# relight deepserver2@/share/datasets/data_synth
# export DISPLAY=:0.0
import os
from relight import relight
from os.path import join

root = "/mnt/data2/work/projects/Samsung/FaceRelighting/lightestim/lightestim/results/render"
dst_root = "/mnt/data2/work/projects/Samsung/FaceRelighting/datageneration/renders/renders_multi"
os.makedirs(dst_root, exist_ok=True)

try:
    relight(root, dst_root, "frame.png", "normals_diffused.png", size=256, world_color=0.17, intensity=4.2, sun_angle=0.785398, blender_name="datagen_lightestim.blend", print_command=True)
except:
    print(f"ERROR executing {image_name}")
