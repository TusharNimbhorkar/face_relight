#Generates SH coefficients from light direction
import os

from utils.utils_SH import SH_basis, euler_to_dir, gen_half_sphere, convert_sh_to_3dul
import glob
import os.path as osp
import csv
import numpy as np
import argparse
import shutil
import cv2
import random
from commons.common_tools import sort_numerically
import matplotlib.pyplot as plt
from utils.utils_data import resize_pil



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="Input Data Directory")
ap.add_argument("-s", "--size", required=False, default=-1, type=int,
	help="Original image size")
ap.add_argument("-p", "--prev_dir", required=True,
	help="Older Data Directory")
ap.add_argument("--no_orig", required=False, action='store_true',
	help="Disables copying of original images")
ap.add_argument("--del_orig", required=False, action='store_true',
	help="Deletes original images")
ap.add_argument("--index_data_dir", required=False, default=None, type=str,
	help="Disables copying of original images")

ap.add_argument("--disable_sun_diameter", required=False, action='store_false',
	help="Deletes original images")
ap.add_argument("--disable_sun_color", required=False, action='store_false',
	help="Deletes original images")
ap.add_argument("--disable_amb_color", required=False, action='store_false',
	help="Deletes original images")
ap.add_argument("--disable_amb_intensity", required=False, action='store_false',
	help="Deletes original images")
args = vars(ap.parse_args())

data_dir = args['input_dir'] #'/home/nedko/face_relight/dbs/example_data'
older_date_dir = args['prev_dir'] #/home/tushar/data2/DPR/train
index_data_dir = args['index_data_dir']
light_info_fname = 'index.txt'
out_sh_fname = 'light_%s_sh.txt'
out_orig_img_fname = 'orig.png'

# orig_img_fname = '%s_05.png'
# orig_sh_fname = '%s_light_05.txt'

orig_img_fname = '%s.png'
orig_sh_fname = 'ori_sh.txt'

orig_size = args['size']
delete_orig_img = args['del_orig']

enable_sun_diameter = args['disable_sun_diameter']
enable_ambient_intensity = args['disable_amb_intensity']
enable_ambient_color = args['disable_amb_color']
enable_sun_color = args['disable_sun_color']

# enable_sun_diameter = True
# enable_ambient_intensity = True
# enable_ambient_color = True
# enable_sun_color = False
props=None

entry_dirs = glob.glob(osp.join(data_dir, '*'))
sort_numerically(entry_dirs)
for entry_dir in entry_dirs:
    print(entry_dir)
    entry_subname = entry_dir.rsplit('/', 1)[-1]

    if index_data_dir is not None:
        index_entry_dir = osp.join(index_data_dir, entry_subname)
    else:
        index_entry_dir = entry_dir
    light_info_path = osp.join(index_entry_dir, light_info_fname)
    with open(light_info_path, 'r') as light_info_file:
        csv_reader = csv.reader(light_info_file, delimiter=',')
        title = next(csv_reader)

        if props is None:
            title[0]=title[0][1:]
            props = {str.strip():i for i, str in enumerate(title)}
            has_sun_diameter = ('sun_angle' in props.keys())
            has_ambient_intensity = ('world_intensity' in props.keys())
            has_ambient_color = ('ambient_r' in props.keys()) & ('ambient_g' in props.keys()) & ('ambient_b' in props.keys())
            has_sun_color = ('sun_r' in props.keys()) & ('sun_g' in props.keys()) & ('sun_b' in props.keys())

            if enable_sun_diameter != has_sun_color or enable_ambient_intensity != has_ambient_intensity or \
                enable_ambient_color != has_ambient_color or enable_sun_color != has_sun_color:

                print("Some of the requested parameters are not available!")

            enable_sun_diameter = enable_sun_diameter & has_sun_diameter
            enable_ambient_intensity = enable_ambient_intensity & has_ambient_intensity
            enable_ambient_color = enable_ambient_color & has_ambient_color
            enable_sun_color = enable_sun_color & has_sun_color

            # print('TEST', props, enable_sun_diameter, enable_ambient_intensity, enable_ambient_color, enable_sun_color)
            # exit(0)

        # print(light_info_file)
        for row in csv_reader:
            light_euler = np.asarray([float(props['light_Y']), float(props['light_Z'])])[...].astype(np.float32)
            light_dir = -euler_to_dir(0, light_euler[0], light_euler[1])[np.newaxis, ...]

            # light_dir = light_dir / np.linalg.norm(light_dir)
            sh_coeffs = SH_basis(light_dir)
            light_intensity_blender = float(props['light_intensity'])
            sh_coeffs[0, 0] = light_intensity_blender - 4.0

            if enable_ambient_intensity:
                ambient_intensity = float(props['world_intensity'])
                sh_coeffs = np.append(sh_coeffs, [[ambient_intensity]], axis=1)

            if enable_ambient_color:
                ambient_intensity_r = float(props['ambient_r'])
                ambient_intensity_g = float(props['ambient_g'])
                ambient_intensity_b = float(props['ambient_b'])
                sh_coeffs = np.append(sh_coeffs, [[ambient_intensity_r]], axis=1)
                sh_coeffs = np.append(sh_coeffs, [[ambient_intensity_g]], axis=1)
                sh_coeffs = np.append(sh_coeffs, [[ambient_intensity_b]], axis=1)

            if enable_sun_diameter:
                sun_diameter = float(props['sun_angle'])
                sh_coeffs = np.append(sh_coeffs, [[sun_diameter]], axis=1)

            if enable_sun_color:
                sun_r = float(props['sun_r'])
                sun_g = float(props['sun_g'])
                sun_b = float(props['sun_b'])
                sh_coeffs = np.append(sh_coeffs, [[sun_r]], axis=1)
                sh_coeffs = np.append(sh_coeffs, [[sun_g]], axis=1)
                sh_coeffs = np.append(sh_coeffs, [[sun_b]], axis=1)

            # shading = gen_half_sphere(sh_coeffs.T)

            img_subname = row[0].split('.')[0]
            out_sh_path = osp.join(entry_dir, out_sh_fname % img_subname)
            # print(sh_coeffs)
            np.savetxt(out_sh_path, sh_coeffs.T, delimiter=',')
        # print()

        orig_img_path = osp.join(older_date_dir, entry_subname, orig_img_fname % entry_subname)
        orig_sh_path = osp.join(older_date_dir, entry_subname, orig_sh_fname) #% entry_subname)
        out_orig_img_path = osp.join(entry_dir, out_orig_img_fname)
        out_orig_sh_path = osp.join(entry_dir, out_sh_fname % 'orig')

        if not args['no_orig']:
            if orig_size > 0:
                # print(orig_img_path)
                orig_img = cv2.imread(orig_img_path)
                # orig_img = cv2.resize(orig_img, (orig_size, orig_size))
                orig_img = resize_pil(orig_img, width=orig_size,height=orig_size)

                cv2.imwrite(out_orig_img_path, orig_img)
            else:
                shutil.copyfile(orig_img_path, out_orig_img_path)
        elif delete_orig_img:
            if osp.exists(out_orig_img_path):
                print('Deleting:', out_orig_img_path)
                os.remove(out_orig_img_path)

        sh = np.loadtxt(orig_sh_path)
        # plt.imshow(gen_half_sphere(sh))
        # plt.show()

        sh_rot = convert_sh_to_3dul(sh)

        sh_rot[0]=0 #8.862269254527579410e-01
        ambient_intensity_blender_orig = 0
        ambient_intensity_blender_orig_color = [1,1,1]
        sun_diameter = 3.2 + random.uniform(-0.5,0)
        sun_color = [1,1,1]

        if enable_ambient_intensity:
            sh_rot = np.append(sh_rot, [ambient_intensity_blender_orig], axis=0)

        if enable_ambient_color:
            sh_rot = np.append(sh_rot, [ambient_intensity_blender_orig_color[0]], axis=0)
            sh_rot = np.append(sh_rot, [ambient_intensity_blender_orig_color[1]], axis=0)
            sh_rot = np.append(sh_rot, [ambient_intensity_blender_orig_color[2]], axis=0)

        if enable_sun_diameter:
            sh_rot = np.append(sh_rot, [sun_diameter], axis=0)

        if enable_sun_color:
            sh_rot = np.append(sh_rot, [sun_color[0]], axis=0)
            sh_rot = np.append(sh_rot, [sun_color[1]], axis=0)
            sh_rot = np.append(sh_rot, [sun_color[2]], axis=0)

        np.savetxt(out_orig_sh_path, sh_rot)
        # shutil.copyfile(orig_sh_path, out_orig_sh_path)
