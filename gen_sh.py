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
import json



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

props_ids=None

def gen_orig_sh(orig_sh_path):
    output_props = {}
    sh = np.loadtxt(orig_sh_path)
    # plt.imshow(gen_half_sphere(sh))
    # plt.show()

    sh_rot = convert_sh_to_3dul(sh)

    output_props['sh'] = (sh_rot.reshape((-1))).tolist()
    output_props['sun_intensity'] = 0

    if enable_ambient_intensity:
        output_props['ambient_intensity'] = 0

    if enable_ambient_color:
        output_props['ambient_color'] = [1,1,1]

    if enable_sun_diameter:
        output_props['sun_diameter'] = 3.2 + random.uniform(-0.5, 0)

    if enable_sun_color:
        output_props['sun_color'] = [1,1,1]

    return output_props



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

        orig_img_path = osp.join(older_date_dir, entry_subname, orig_img_fname % entry_subname)
        orig_sh_path = osp.join(older_date_dir, entry_subname, orig_sh_fname) #% entry_subname)
        out_orig_img_path = osp.join(entry_dir, out_orig_img_fname)
        out_orig_sh_path = osp.join(entry_dir, out_sh_fname % 'orig')

        if props_ids is None:
            title[0]=title[0][1:]
            props_ids = {str.strip():i for i, str in enumerate(title)}
            has_sun_diameter = ('sun_angle' in props_ids.keys())
            has_ambient_intensity = ('world_intensity' in props_ids.keys())
            has_ambient_color = ('ambient_r' in props_ids.keys()) & ('ambient_g' in props_ids.keys()) & ('ambient_b' in props_ids.keys())
            has_sun_color = ('sun_r' in props_ids.keys()) & ('sun_g' in props_ids.keys()) & ('sun_b' in props_ids.keys())

            if enable_sun_diameter != has_sun_color or enable_ambient_intensity != has_ambient_intensity or \
                enable_ambient_color != has_ambient_color or enable_sun_color != has_sun_color:

                print("Some of the requested parameters are not available!")

            enable_sun_diameter = enable_sun_diameter & has_sun_diameter
            enable_ambient_intensity = enable_ambient_intensity & has_ambient_intensity
            enable_ambient_color = enable_ambient_color & has_ambient_color
            enable_sun_color = enable_sun_color & has_sun_color

        for row in csv_reader:
            output_props = {}
            input_props = {}
            for (key, value) in props_ids.items():
                input_props[key] = row[value]

            img_subname = input_props['name'].split('.')[0]

            if 'orig' in img_subname:
                output_props = gen_orig_sh(orig_sh_path)

                if enable_ambient_color:
                    ambient_intensity_r = float(input_props['ambient_r'])
                    ambient_intensity_g = float(input_props['ambient_g'])
                    ambient_intensity_b = float(input_props['ambient_b'])
                    output_props['ambient_color'] = [ambient_intensity_r, ambient_intensity_g, ambient_intensity_b]

            else:
                light_euler = np.asarray([float(input_props['light_Y']), float(input_props['light_Z'])])[...].astype(np.float32)
                light_dir = -euler_to_dir(0, light_euler[0], light_euler[1])[np.newaxis, ...]

                sh_coeffs = SH_basis(light_dir)
                output_props['sh'] = (sh_coeffs.T).reshape((-1)).tolist()

                light_intensity_blender = float(input_props['light_intensity'])
                output_props['sun_intensity'] = light_intensity_blender - 4.0

                if enable_ambient_intensity:
                    ambient_intensity = float(input_props['world_intensity'])
                    output_props['ambient_intensity'] = ambient_intensity

                if enable_ambient_color:
                    ambient_intensity_r = float(input_props['ambient_r'])
                    ambient_intensity_g = float(input_props['ambient_g'])
                    ambient_intensity_b = float(input_props['ambient_b'])
                    output_props['ambient_color'] = [ambient_intensity_r, ambient_intensity_g, ambient_intensity_b]

                if enable_sun_diameter:
                    sun_diameter = float(input_props['sun_angle'])
                    output_props['sun_diameter'] = sun_diameter

                if enable_sun_color:
                    sun_r = float(input_props['sun_r'])
                    sun_g = float(input_props['sun_g'])
                    sun_b = float(input_props['sun_b'])
                    output_props['sun_color'] = [sun_r, sun_g, sun_b]


            out_sh_path = osp.join(entry_dir, out_sh_fname % img_subname)

            with open(out_sh_path, 'w') as output_file:
                json.dump(output_props, output_file, indent=4)
        # print()


        if not args['no_orig']:
            if orig_size > 0:
                orig_img = cv2.imread(orig_img_path)
                orig_img = resize_pil(orig_img, width=orig_size,height=orig_size)

                cv2.imwrite(out_orig_img_path, orig_img)
            else:
                shutil.copyfile(orig_img_path, out_orig_img_path)
        elif delete_orig_img:
            if osp.exists(out_orig_img_path):
                print('Deleting:', out_orig_img_path)
                os.remove(out_orig_img_path)

        output_props = gen_orig_sh(orig_sh_path)

        with open(out_orig_sh_path, 'w') as output_file:
            json.dump(output_props, output_file, indent=4)
        # shutil.copyfile(orig_sh_path, out_orig_sh_path)
