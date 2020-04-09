#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

def vis_parsing_maps(im, parsing_anno, stride,h=None,w=None):

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # MASK
    vis_parsing_anno = cv2.resize(vis_parsing_anno,(w,h))
    cv2.imwrite('segment.png',vis_parsing_anno)
    # pri


    # before
    '''
    vis_parsing_anno[vis_parsing_anno==16]=0
    # vis_parsing_anno[vis_parsing_anno==16]=0
    vis_parsing_anno[vis_parsing_anno>0]=1
    # cv2.imwrite('segment.png',vis_parsing_anno)
    '''
    # after

    # only face skin
    '''
    vis_parsing_anno[vis_parsing_anno==16]=0
    vis_parsing_anno[vis_parsing_anno==17]=0
    vis_parsing_anno[vis_parsing_anno == 14] = 0
    vis_parsing_anno[vis_parsing_anno>0]=255
    '''

    # face+hair
    '''
    vis_parsing_anno[vis_parsing_anno == 16] = 0
    # vis_parsing_anno[vis_parsing_anno==17]=0
    vis_parsing_anno[vis_parsing_anno == 14] = 0
    vis_parsing_anno[vis_parsing_anno > 0] = 255
    '''



    return vis_parsing_anno


def evaluate(img):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('cp', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        
        w, h = img.size
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        output_img = vis_parsing_maps(image, parsing, stride=1,h=h,w=w)
    return output_img


if __name__ == "__main__":
    image_path = 'data/ori.jpg'
    print('it runs')
    img = Image.open(osp.join(image_path))
    segment = evaluate(img)


