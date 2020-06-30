import os
from os.path import join
from scipy.io import loadmat
import random
import cv2
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
greyball_input = join(dir_path, './real_illum_11346.mat')
greyball = loadmat(greyball_input)
ambient_color = greyball['real_rgb'].tolist()

def get_ambient_color():
    rgb_color = random.choice(ambient_color)
    rgb_color = np.array(rgb_color).reshape((1,1,3))*255
    rgb_color = rgb_color.astype(np.uint8)
    print(rgb_color)

    hsv = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    hsv_orig = np.copy(hsv)
    hsv_orig[:,:,2] = 255
    rgb_orig = cv2.cvtColor(hsv_orig, cv2.COLOR_HSV2RGB)/255

    hsv_rel = np.copy(hsv)
    hsv_rel[:, :, 2] = int(random.uniform(0.2, 0.9)*255)
    rgb_rel = cv2.cvtColor(hsv_rel, cv2.COLOR_HSV2RGB)/255

    # intensity = random.uniform(3,6)

    return rgb_orig, rgb_rel

# print(get_ambient_color())

def get_sun_color():
    return random.choice(ambient_color)
