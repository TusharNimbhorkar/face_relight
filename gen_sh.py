#Generates SH coefficients from light direction

from utils.utils_SH import SH_basis, euler_to_dir, gen_half_sphere
import glob
import os.path as osp
import csv
import numpy as np
import argparse
import shutil
from commons.common_tools import sort_numerically
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="Input Data Directory")
ap.add_argument("-p", "--prev_dir", required=True,
	help="Older Data Directory")
args = vars(ap.parse_args())

data_dir = args['input_dir'] #'/home/nedko/face_relight/dbs/example_data'
older_date_dir = args['prev_dir'] #/home/tushar/data2/DPR/train
light_info_fname = 'index.txt'
out_sh_fname = 'light_%s_sh.txt'

orig_img_fname = '%s_05.png'
orig_sh_fname = '%s_light_05.txt'
out_orig_img_fname = 'orig.png'

entry_dirs = glob.glob(osp.join(data_dir, '*'))
sort_numerically(entry_dirs)
for entry_dir in entry_dirs:
    print(entry_dir)
    light_info_path = osp.join(entry_dir, light_info_fname)
    with open(light_info_path, 'r') as light_info_file:
        csv_reader = csv.reader(light_info_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            light_euler = np.asarray([float(val) for val in row[1:3]])[...].astype(np.float32)
            light_dir = -euler_to_dir(0, light_euler[0], light_euler[1])[np.newaxis, ...]
            # light_dir = light_dir / np.linalg.norm(light_dir)
            sh_coeffs = SH_basis(light_dir)

            # shading = gen_half_sphere(sh_coeffs.T)

            img_subname = row[0].split('.')[0]
            out_sh_path = osp.join(entry_dir, out_sh_fname % img_subname)
            np.savetxt(out_sh_path, sh_coeffs.T, delimiter=',')

        entry_subname = entry_dir.rsplit('/',1)[-1]
        orig_img_path = osp.join(older_date_dir, entry_subname, orig_img_fname % entry_subname)
        orig_sh_path = osp.join(older_date_dir, entry_subname, orig_sh_fname % entry_subname)
        out_orig_img_path = osp.join(entry_dir, out_orig_img_fname)
        out_orig_sh_path = osp.join(entry_dir, out_sh_fname % 'orig')

        shutil.copyfile(orig_img_path, out_orig_img_path)
        shutil.copyfile(orig_sh_path, out_orig_sh_path)
