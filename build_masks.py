import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os.path as osp
import os
from commons.common_tools import Logger, BColors
import pickle

log = Logger('Overexposure Detector')

src_mask_dir = '/home/nedko/face_relight/dbs/CelebAMask-HQ/all/'
src_mask_subfnames = [
                    "%05d_l_brow.png",
                      "%05d_l_eye.png",
                      "%05d_l_lip.png",
                      # "%05d_nose.png",
                      "%05d_r_brow.png",
                      "%05d_r_eye.png",
                      # "%05d_skin.png",
                      "%05d_u_lip.png",
                      "%05d_mouth.png",
                        ]
src_mask_skin_subfname = "%05d_skin.png"

output_dir = "/home/nedko/face_relight/dbs/CelebAMask-HQ/composed/"
os.makedirs(output_dir, exist_ok=True)

def invert(mask):
    return np.abs(mask-1)

def build_mask(id, src_mask_dir, src_mask_subfnames):
    submask_path = osp.join(src_mask_dir, src_mask_skin_subfname % id)
    mask = np.asarray(cv2.imread(submask_path, cv2.IMREAD_GRAYSCALE))

    for subfname in src_mask_subfnames:
        submask_path = osp.join(src_mask_dir, subfname % id)
        if osp.exists(submask_path):
            submask = np.asarray(cv2.imread(submask_path, cv2.IMREAD_GRAYSCALE))
            # submask = invert(submask)
            if mask is None:
                mask = submask
            else:
                mask = mask-submask
        else:
            log.e('File not found: ', submask_path)

    mask = np.clip(mask, 0, 255)
    # mask[mask>0]=1
    # plt.imshow(mask*255)
    # plt.show()
    return mask

# out = open(output_path, 'wb')
max_id = 29999

for id in range(27808,max_id+1):
    output_path = osp.join(output_dir, "%05d.png" % id)
    mask = build_mask(id, src_mask_dir, src_mask_subfnames)
    mask = mask[:,:,np.newaxis]
    cv2.imwrite(output_path, (mask).astype(np.uint8))

    log.i('Processed ', id)
