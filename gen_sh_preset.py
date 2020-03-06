# Generates SH presets

import numpy as np
import argparse
from utils.utils_SH import euler_to_dir, SH_basis, show_half_sphere, gen_half_sphere
import os.path as osp
import os
import cv2


preset_storage_dir = './test_data/sh_presets'
sh_fname_template = 'rotate_light_%02d.txt'
half_sphere_fname_template = 'rotate_light_%02d.png'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pattern_name", required=True,
	help="Name of the preset pattern to generate")
args = vars(ap.parse_args())

if args['pattern_name'] == 'horizontal':
    yaw_range = list(range(-90,90,2))
    pitch_range = [0]*len(yaw_range)
    roll_range = [0]*len(yaw_range)
elif args['pattern_name'] == 'round':
    roll_range = list(range(0,360,3))
    yaw_range = [45]*len(roll_range)
    pitch_range = [0]*len(roll_range)
else:
    raise NotImplementedError()

preset_dir = osp.join(preset_storage_dir, args['pattern_name'])
os.makedirs(preset_dir, exist_ok=True)
out_sh_path_template = osp.join(preset_dir, sh_fname_template)
out_half_sphere_path_template = osp.join(preset_dir, half_sphere_fname_template)

for i, (yaw, pitch, roll) in enumerate(zip(yaw_range, pitch_range, roll_range)):
    light_dir = -euler_to_dir(np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll))
    light_dir = light_dir / np.linalg.norm(light_dir)
    sh_coeffs = SH_basis(light_dir[np.newaxis, ...])
    np.savetxt(out_sh_path_template % i, sh_coeffs)

    # show_half_sphere(sh_coeffs.T)
    half_sphere_img = gen_half_sphere(sh_coeffs.T)
    cv2.imwrite(out_half_sphere_path_template % i, half_sphere_img)