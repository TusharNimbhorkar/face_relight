#!/usr/bin/python
# relight deepserver2@/share/datasets/data_synth
# export DISPLAY=:0.0
import os
from relight import relight
from os.path import join

# ~ root = "/mnt/data2/users/morris/data_synth"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/10k_v1"
# ~ root = "/mnt/data2/users/tushar/rendering_pipeline/stylegan_normals"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/stylegan_normals_v1"

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal.png", size=256, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal_faceRegion_faceBoundary_extend.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")


# other 20K from DPR
# ~ root = "/mnt/data2/users/tushar/DPR_normals_10kto30k"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/20k_v1"

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal_faceRegion_faceBoundary_extend.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")


# ~ root = "/mnt/data2/users/morris/data_synth"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/10k_v2"

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal_faceRegion_faceBoundary_extend.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")


# other 20K from DPR
# ~ root = "/mnt/data2/users/tushar/DPR_normals_10kto30k"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/20k_v2"

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal_faceRegion_faceBoundary_extend.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")

# ~ root = "/home/tushar/data2/rendering_pipeline/stylegan_final_30k"
# ~ dst_root = "/mnt/data2/users/morris/data_relight/stylegan_final_30k_highres"

# ~ try:
    # ~ relight(root, dst_root, "frame.png", "full_normal.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", print_command=True) # 0.6108652381980153
# ~ except:
    # ~ print(f"ERROR executing blender")

root = "/mnt/data1/users/morris/Relight/data/stylegan_final_30k"
dst_root = "/mnt/data1/users/morris/Relight/results/stylegan_final_30k_ambient_noangle"
# ~ dst_root = "/opt/stylegan_final_30k_ambient"

try:
    relight(root, dst_root, "frame.png", "full_normal.png", size=1024, world_color=0.28, intensity=4.0, sun_angle=1.0471975511965976, blender_name="datagen.blend", relight_name="relight_ambient_color.py", print_command=True) # 0.6108652381980153    
    print("granting access")
    os.system(f"chmod -R 777 \"{dst_root}\"")
    print("DONE!")
except:
    print(f"ERROR executing blender")
    
