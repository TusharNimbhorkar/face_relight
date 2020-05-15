'''
Stores common code between models and data loaders
'''

import cv2
import numpy as np


def img_to_input(img, input_mode, transform=None):
    '''
    Converts a numpy RGB image array into network input array
    :param img: RGB image numpy array
    :return: input array ready for network input
    '''

    if input_mode == 'L':
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        input_L = img_lab[:, :, 0]
        input_L_norm = input_L.astype(np.float32) / 255.0
        input_L_norm = input_L_norm.transpose((0, 1))
        input = input_L_norm[..., None]
    elif input_mode == 'RGB':
        input = img.astype(np.float32) / 255.0
    elif input_mode == 'LAB':
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        input = img_lab.astype(np.float32) / 255.0

    if transform is not None:
        input = transform(input)

    # print('TEST shape', input.shape)

    return input

def output_to_img(output_img, input_mode, input_img = None):
    output_img = output_img[0].cpu().data.numpy()
    output_img = output_img.transpose((1, 2, 0))
    output_img = np.squeeze(output_img)  # *1.45
    output_img = (output_img * 255.0).astype(np.uint8)

    if input_mode == 'L':
        if input_img is None:
            raise ValueError()

        img_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
        img_lab[:, :, 0] = output_img
        output_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    elif input_mode == 'RGB':
        pass
    elif input_mode == 'LAB':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)

    return output_img