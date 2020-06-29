import argparse
import multiprocessing
import shutil

from commons.common_tools import is_image_file
from utils.utils_data import InputProcessor
import glob
import os.path as osp
import os
import cv2
import numpy as np
from commons.common_tools import sort_numerically

# # orig_img_dir = '/home/tushar/data2/rendering_pipeline/stylegan_final_30k'
# input_data_path = "/home/nedko/face_relight/dbs/data/stylegan_v0/v0.3.1_1024_ambient_0p04to0p4"
# output_data_dir = "dbs/data/stylegan_v0/v0.4.1_1024_int_ambient_side_offset_crop"
# face_data_dir = "/home/nedko/face_relight/dbs/data/stylegan_v0/face_data"


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="Input Data Directory")
ap.add_argument("-p", "--prev_dir", required=True,
                help="Original Images Directory")
ap.add_argument("-o", "--output_dir", required=True,
                help="Output Directory")
ap.add_argument("-f", "--face_data_dir", required=True,
                help="Disables copying of original images")
ap.add_argument("-n", "--first_n", required=False, type=int, default=0,
                help="Number of images to crop")
ap.add_argument("--enable_overwrite", required=False, action="store_true",
                help="If a cropped image already exists, overwrite it")
args = vars(ap.parse_args())

input_data_path = args['input_dir']
output_data_dir = args['output_dir']
face_data_dir = args['face_data_dir']
orig_dir = args['prev_dir']
side_offset = True
first_n = args['first_n']

n_threads = 6
chunk_size = 30
# n_files_in_folder = 6
enable_overwrite = args['enable_overwrite']

model_dir = osp.abspath('./demo/data/model')
input_processor = InputProcessor(model_dir)
data_id_paths = glob.glob(osp.join(input_data_path, '*'))
sort_numerically(data_id_paths)

if first_n !=0:
    data_id_paths = data_id_paths[:first_n]

os.makedirs(face_data_dir, exist_ok=True)

def crop(data_id_path):
    id_dirname = data_id_path.rsplit('/', 1)[-1]
    # print(id_dirname)
    has_crop_in_dir = False

    img_orig_fname = "%s.png" % id_dirname  # "orig" #id_dirname
    img_path_orig = osp.join(orig_dir, id_dirname, img_orig_fname)
    # img_orig_fname = "orig.png"
    # img_path_orig = osp.join(data_id_path, img_orig_fname)

    output_id_path = osp.join(output_data_dir, id_dirname)
    os.makedirs(output_id_path, exist_ok=True)
    face_data_id_path = osp.join(face_data_dir, id_dirname + '.npz')
    output_img_path = osp.join(output_data_dir, id_dirname, "orig.png")
    # print(img_path_orig)

    # if osp.exists(output_img_path) and len(os.listdir(output_data_dir)) >= n_files_in_folder:
    #     return

    # Coopy all non image files without change and collect all image paths
    img_paths = []
    for file_path in glob.glob(osp.join(data_id_path, '*')):
        if osp.isdir(file_path):
            pass
        elif is_image_file(file_path):
            img_paths.append(file_path)
        else:
            fname = file_path.rsplit('/', 1)[-1]
            shutil.copy(file_path, osp.join(output_id_path, fname))

    # print('TEST ORIG', img_path_orig)
    img_orig = cv2.imread(img_path_orig)

    if osp.exists(face_data_id_path):
        face_data = np.load(face_data_id_path, allow_pickle=True)
        face_data = [face_data[key] for key in ["rects", "scores", "idx", "shape"]]
    else:
        face_data = input_processor.get_face_data(img_orig)
        rects, scores, idx, shape = face_data
        np.savez(face_data_id_path, rects=rects, scores=scores, idx=idx, shape=shape)

    if not osp.exists(output_img_path) or enable_overwrite:
        has_crop_in_dir = True
        if side_offset:
            img_orig_proc = input_processor.process_side_offset(img_orig, face_data)[0]
        else:
            img_orig_proc = input_processor.process(img_orig, face_data)[0]

        cv2.imwrite(output_img_path, img_orig_proc)

    for img_path in img_paths:
        if img_path == img_path_orig:
            continue

        img_fname = img_path.rsplit('/', 1)[-1]
        output_img_path = osp.join(output_data_dir, id_dirname, img_fname)

        if osp.exists(output_img_path) and not enable_overwrite:
            continue

        has_crop_in_dir = True
        img = cv2.imread(img_path)
        if side_offset:
            img_proc = input_processor.process_side_offset(img, face_data=face_data)[0]
        else:
            img_proc = input_processor.process(img, face_data=face_data)[0]

        cv2.imwrite(output_img_path, img_proc)

    if has_crop_in_dir:
        print('Cropping to: ', output_img_path)


pool = multiprocessing.Pool(n_threads)

i = 1
for x in pool.imap_unordered(crop, data_id_paths, chunksize=chunk_size):
    if i % 100 == 0:
        print("Completed: ", i)
    i += 1
# for data_id_path in data_id_paths:
