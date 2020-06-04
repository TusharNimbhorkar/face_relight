import shutil

from commons.common_tools import is_image_file
from utils.utils_data import InputProcessor
import glob
import os.path as osp
import os
import cv2
from commons.common_tools import sort_numerically

orig_img_dir = '/home/tushar/data2/rendering_pipeline/stylegan_final_30k'
input_data_path = "/home/nedko/face_relight/dbs/data/stylegan_v0/v0.2_1024"
output_data_dir = "/home/nedko/face_relight/dbs/data/stylegan_v0/v0.2_1024_10k_crop"
first_n = 10000

model_dir = osp.abspath('./demo/data/model')
input_processor = InputProcessor(model_dir)
data_id_paths = glob.glob(osp.join(input_data_path, '*'))[:first_n]
sort_numerically(data_id_paths)

for data_id_path in data_id_paths:
    id_dirname = data_id_path.rsplit('/', 1)[-1]

    img_orig_fname = "%s.png" % id_dirname
    img_path_orig = osp.join(orig_img_dir, id_dirname, img_orig_fname)
    # img_orig_fname = "orig.png"
    # img_path_orig = osp.join(data_id_path, img_orig_fname)

    output_id_path = osp.join(output_data_dir, id_dirname)
    os.makedirs(output_id_path, exist_ok=True)
    output_img_path = osp.join(output_data_dir, id_dirname, "orig.png")
    print(img_path_orig)

    #Coopy all non image files without change and collect all image paths
    img_paths = []
    for file_path in glob.glob(osp.join(data_id_path, '*')):
        if is_image_file(file_path):
            img_paths.append(file_path)
        else:
            fname = file_path.rsplit('/', 1)[-1]
            shutil.copy(file_path, osp.join(output_id_path, fname))

    img_orig = cv2.imread(img_path_orig)
    det_data = input_processor.get_face_det(img_orig)
    img_orig_proc = input_processor.process(img_orig, det_data)[0]
    cv2.imwrite(output_img_path, img_orig_proc)

    for img_path in img_paths:
        if img_path == img_path_orig:
            continue

        img_fname = img_path.rsplit('/',1)[-1]
        output_img_path = osp.join(output_data_dir, id_dirname, img_fname)
        img = cv2.imread(img_path)
        img_proc = input_processor.process(img, det_data)[0]
        cv2.imwrite(output_img_path, img_proc)
