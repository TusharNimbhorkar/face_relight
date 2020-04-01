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
    X_range = list(range(-90, 90, 2))
    Y_range = [90] * len(X_range)
    Z_range = [0] * len(X_range)
elif args['pattern_name'] == 'round':
    Z_range = list(range(0, 360, 3))
    X_range = [45] * len(Z_range)
    Y_range = [0] * len(Z_range)
elif args['pattern_name'] == 'vertical':
    vert_range = list(range(90-60, 90+60, 2))
    length = len(vert_range)
    zeros = [0] * length

    X_range = zeros
    Y_range = vert_range#list(range(-90, 90, 2))
    Z_range = zeros#[0] * len(X_range)
else:
    raise NotImplementedError()

preset_dir = osp.join(preset_storage_dir, args['pattern_name'])
os.makedirs(preset_dir, exist_ok=True)
out_sh_path_template = osp.join(preset_dir, sh_fname_template)
out_half_sphere_path_template = osp.join(preset_dir, half_sphere_fname_template)

for i, (X, Y, Z) in enumerate(zip(X_range, Y_range, Z_range)):
    light_dir = euler_to_dir(np.deg2rad(X), np.deg2rad(Y), np.deg2rad(Z))
    # light_dir = light_dir / np.linalg.norm(light_dir)
    print(light_dir)
    light_dir = -light_dir
    sh_coeffs = SH_basis(light_dir[np.newaxis, ...])
    np.savetxt(out_sh_path_template % i, sh_coeffs)

    # show_half_sphere(sh_coeffs.T)
    half_sphere_img = gen_half_sphere(sh_coeffs.T)
    cv2.imwrite(out_half_sphere_path_template % i, half_sphere_img)