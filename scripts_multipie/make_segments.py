import cv2
import matplotlib.pyplot as plt
import os,glob
import numpy as np
# local
# mask_dir_folder = '/home/tushar/bayer2rgb/mask/0'
# folders_DPR = os.listdir('/home/tushar/DPR_data/skel/train')
# save_folder = '/home/tushar/DPR_data/skel/segments'
# server
mask_dir_folder = '/home/tushar/data2/DPR/CelebAMask-HQ/combined'
folders_DPR = os.listdir('/home/tushar/data2/DPR/train')
save_folder = '/home/tushar/data2/DPR/segments'




if not os.path.exists(save_folder):
    os.makedirs(save_folder)

files = sorted(os.listdir(mask_dir_folder))

temp_files = []
for  kk in files:
    temp_files.append(kk.split('_')[0])
temp_files = list(set(temp_files))

os.chdir(mask_dir_folder)

for tmp in temp_files:
    temp_list = []
    for f in glob.glob(tmp+'*'):
        temp_list.append(f)
    title = 'imgHQ' + temp_list[0].split('_')[0]
    if title in folders_DPR:
        init_img = cv2.imread(os.path.join(mask_dir_folder, temp_list[0]))
        temp_list = temp_list[1:]
        for k in temp_list:
            seg_img = cv2.imread(os.path.join(mask_dir_folder, k))
            init_img = init_img + seg_img

        init_img[init_img > 0] = 255
        dir_save_segment = os.path.join(save_folder, title)
        if not os.path.exists(dir_save_segment):
            os.makedirs(dir_save_segment)
        save_seg_path = os.path.join(dir_save_segment, title + '.png')
        cv2.imwrite(save_seg_path, init_img)
        # print(k)



print()
# for j in range(0, len(files), 11):
#     temp_list = files[j:j + 11]
#     title = 'imgHQ' + temp_list[0].split('_')[0]
#     if title in folders_DPR:
#         init_img = cv2.imread(os.path.join(mask_dir_folder, temp_list[0]))
#         temp_list=temp_list[1:]
#         for k in temp_list:
#             seg_img = cv2.imread(os.path.join(mask_dir_folder, k))
#             init_img=init_img+seg_img
#
#         init_img[init_img>0]=255
#         dir_save_segment = os.path.join(save_folder,title)
#         if not os.path.exists(dir_save_segment):
#             os.makedirs(dir_save_segment)
#         save_seg_path = os.path.join(dir_save_segment,title+'.png')
#         cv2.imwrite(save_seg_path,init_img)
#         print(k)
#
#
#
# # im1 = cv2.imread('/home/tushar/bayer2rgb/mask/00000_hair.png')
# # im2 = cv2.imread('/home/tushar/DPR_data/skel/train/imgHQ00000/imgHQ00000_05.png')
# # print()

#
# im1 = cv2.resize(cv2.imread(os.path.join(dpr_dir,'train','imgHQ00000','imgHQ00000_05.png')),(512,512))
# im2 = cv2.imread(os.path.join(dpr_dir,'segments','imgHQ00000','imgHQ00000.png'))
# im2[im2>0]=1
# plt.imshow(np.multiply(im1,im2))
# plt.show()