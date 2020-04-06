from data.base_dataset import BaseDataset
from .lightDPR7_dataset import lightDPR7Dataset
import numpy as np

class lightDPR7RGBDataset(lightDPR7Dataset):
    '''
    Use RGB input images with segmentation for the DPR dataset
    '''
    def __init__(self, opt):
        super(lightDPR7RGBDataset, self).__init__(opt)

    def _img_to_input(self, img):
        '''
        Converts a numpy RGB image array into a normalized RGB  tensor
        :param img: RGB image numpy array
        :return: RGB normalized tensor
        '''
        input_norm = img.astype(np.float32) / 255.0
        input = self.transform_A(input_norm)

        return input

    def name(self):
        return 'lightDPR7RGBDataset'
