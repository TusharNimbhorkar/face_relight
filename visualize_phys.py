#Concatenate morris output files together

import glob
import cv2
import os.path as osp
import os
import numpy as np

src_path = '/mnt/data/work/3du/Face Relighting/phys_test/2'
tgt_path = '/mnt/data/work/3du/Face Relighting/phys_test/out'

compared_paths = glob.glob(osp.join(src_path,'*'))

id_paths = glob.glob(osp.join(compared_paths[0],'*'))

print(id_paths)
for id_path in id_paths:
    id = id_path.rsplit('/',1)[-1]
    img_paths = glob.glob(osp.join(id_path,'*'))

    out_id_path = osp.join(tgt_path, id)
    os.makedirs(out_id_path, exist_ok=True)
    for img_path in img_paths:
        imgs = []
        img_fname = img_path.rsplit('/',1)[-1]
        for compared_path in compared_paths:
            in_path = osp.join(compared_path,id, img_fname)
            imgs.append(cv2.imread(in_path))

        concat_img = np.concatenate(imgs, axis=1)
        out_path = osp.join(out_id_path, img_fname)
        cv2.imwrite(out_path, concat_img)