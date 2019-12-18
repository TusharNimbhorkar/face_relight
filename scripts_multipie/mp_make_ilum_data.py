import os
import cv2
import shutil
# save_multipie_dir = '/home/tushar/face_relight/multipie/'
# frames_dir = '/home/tushar/face_relight/subset/all'

save_multipie_dir = '/home/tushar/data2/face_rel_multipie/'
frames_dir = '/home/tushar/data2/MULTIPIE/MultiPIE/front'
frames_all = os.listdir(frames_dir)
count = len(frames_all)

if not os.path.exists(save_multipie_dir):
    os.makedirs(save_multipie_dir)


for i in sorted(frames_all):

    new_folder_name = i[:-7]
    new_file_name = i

    new_path_folder = os.path.join(save_multipie_dir,new_folder_name)

    if not os.path.exists(new_path_folder):
        os.makedirs(new_path_folder)

    shutil.copy(os.path.join(frames_dir,i) , os.path.join(new_path_folder,new_file_name))
    print(count)
    count=count-1
