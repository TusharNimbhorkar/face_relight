import cv2
import os
import os.path as osp
from shutil import copy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--size", required=True,
                help="Target Size")
ap.add_argument("-i", "--input", required=True,
                help="Input Directory")
ap.add_argument("-o", "--output", required=True,
                help="Output Directory")
ap.add_argument("--no_segments", required=False, action="store_true",
                help="Don't process segments")
ap.add_argument("--no_real", required=False, action="store_true",
                help="Don't process real")
ap.add_argument("--no_synth", required=False, action="store_true",
                help="Don't process synthetic")
args = vars(ap.parse_args())

dpr_og_dir= args["input"] #'/mnt/data2/users/morris/data_synth_relight_1024_softness60_20k'
# dpr_og_dir = '/home/tushar/DPR_data/skel'
real_im = 'real_im/'
train = 'train/'
segments = 'segments/'

resize_size = (int(args['size']), int(args['size'])) #(512, 512)
new_dpr_dir = args['output'] #'./dbs/3dulight_v0.8_512'# + str(resize_size[0])

# create directories
if not os.path.exists(new_dpr_dir):
    os.makedirs(new_dpr_dir)

new_segments_dir = os.path.join(new_dpr_dir, segments)
new_train_dir = os.path.join(new_dpr_dir, train)
# real_im
new_real_im_dir = os.path.join(new_dpr_dir, real_im)

if not args['no_segments']:
        if not os.path.exists(new_segments_dir):
            os.makedirs(new_segments_dir)

if not args['no_synth']:
    if not os.path.exists(new_train_dir):
        os.makedirs(new_train_dir)

objs = sorted(os.listdir(dpr_og_dir))

for obj in objs:
    # make_dirs_in_segment_in_train
    obj_train_dir = osp.join(new_train_dir, obj)
    obj_seg_dir = osp.join(new_segments_dir, obj)
    if not os.path.exists(obj_train_dir):
        os.makedirs(obj_train_dir)
    if not os.path.exists(obj_seg_dir):
        os.makedirs(obj_seg_dir)

    if not args['no_segments']:
        segment_im_read = cv2.imread(os.path.join(dpr_og_dir,segments,obj,obj+'.png'))
        segment_im_save_path = os.path.join(obj_seg_dir,obj+'.png')
        cv2.imwrite(segment_im_save_path,cv2.resize(segment_im_read,resize_size))

    if not args['no_synth']:
        #read and save the other files.
        path_rel_obj = osp.join(dpr_og_dir,obj)
        rel_files = sorted(os.listdir(path_rel_obj))

        for file in rel_files:
            if file[-3:]=='txt':
                # print(file)
                src_path_sh = osp.join(path_rel_obj,file)
                dst_path_sh = osp.join(obj_train_dir,file)
                copy(src_path_sh,dst_path_sh)
            else:
                src_path_img = osp.join(path_rel_obj, file)
                im1 = cv2.imread(src_path_img)
                im1_r = cv2.resize(im1,resize_size)
                dst_path_img = osp.join(obj_train_dir,file)
                cv2.imwrite(dst_path_img,im1_r)

if not args['no_real']:
    if not os.path.exists(new_real_im_dir):
        os.makedirs(new_real_im_dir)

    # real_im_resize
    real_im_dir = osp.join(dpr_og_dir,real_im)
    objs_real = sorted(os.listdir(real_im_dir))

    for im_p in objs_real:
        im_r = cv2.imread(osp.join(real_im_dir,im_p))
        im_r = cv2.resize(im_r,resize_size)
        save_real_im = osp.join(new_real_im_dir,im_p)
        cv2.imwrite(save_real_im,im_r)
