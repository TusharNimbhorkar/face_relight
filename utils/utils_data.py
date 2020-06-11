import cv2
import dlib
import numpy as np
import os.path as osp
import torch
from PIL import Image
from torchvision.transforms import transforms

from demo_segment.model import BiSeNet
from matplotlib import pyplot as plt

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def resize_pil(image, width=None, height=None, inter=Image.LANCZOS):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = np.array(
        Image.fromarray(image.astype(np.uint8)).resize(dim, resample=Image.LANCZOS))
    # return the resized image
    return resized


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

class SegmentGenerator:

    def __init__(self, model_dir, device='cuda'):

        n_classes = 19
        self.segment_model = BiSeNet(n_classes=n_classes)
        self.segment_model.to(device)
        segment_model_path = osp.join(model_dir, 'face_parsing.pth')
        self.segment_model.load_state_dict(torch.load(segment_model_path))
        self.segment_model.eval()

        self.device = device
        self.segment_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _vis_parsing_maps(self, im, parsing_anno, stride, h=None, w=None):
        im = np.array(im)
        alpha_2 = np.zeros((h, w, 3))
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        # MASK
        vis_parsing_anno = cv2.resize(vis_parsing_anno, (w, h))
        vis_parsing_anno[vis_parsing_anno == 16] = 0
        # vis_parsing_anno[vis_parsing_anno==17]=0
        vis_parsing_anno[vis_parsing_anno == 14] = 0
        vis_parsing_anno[vis_parsing_anno > 0] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closing = cv2.morphologyEx(vis_parsing_anno, cv2.MORPH_CLOSE, kernel)
        closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        new_img = np.zeros_like(closing)  # step 1
        for val in np.unique(closing)[1:]:  # step 2
            mask = np.uint8(closing == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val

        vis_parsing_anno = new_img.copy()

        # alpha_2 = cv2.imread(segment_path_ear)
        alpha_2[:, :, 0] = np.copy(vis_parsing_anno)
        alpha_2[:, :, 1] = np.copy(vis_parsing_anno)
        alpha_2[:, :, 2] = np.copy(vis_parsing_anno)
        kernel = np.ones((10, 10), np.uint8)
        alpha_2 = cv2.erode(alpha_2, kernel, iterations=1)
        alpha_2 = cv2.GaussianBlur(alpha_2, (29, 29), 15, 15)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha_2 = alpha_2.astype(float) / 255
        return alpha_2

    def segment(self, img_):
        with torch.no_grad():
            h, w, _ = img_.shape
            image = cv2.resize(img_, (512, 512), interpolation=cv2.INTER_AREA)
            img = self.segment_norm(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            out = self.segment_model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            output_img = self._vis_parsing_maps(image, parsing, stride=1,h=h,w=w)
        return output_img

class InputProcessor:

    def __init__(self, model_dir, enable_segment=False, device='cuda'):
        self.detector = dlib.get_frontal_face_detector()
        lmarks_model_path = osp.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(lmarks_model_path)
        self.enable_segment = enable_segment

        if enable_segment:
            self.seg_gen = SegmentGenerator(model_dir, device)
        else:
            self.seg_gen = None

        self.R_90 = R(np.deg2rad(90))

    def get_face_data(self, img):

        img = np.array(img)
        orig_size = img.shape

        if np.max(img.shape[:2]) > 1024:
            if img.shape[0] < img.shape[1]:
                img_res = resize_pil(img, width=1024)
            else:
                img_res = resize_pil(img, height=1024)
        else:
            img_res = img

        resize_ratio = orig_size[0] / img_res.shape[0]
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        rects, scores, idx = self.detector.run(gray, 1, -1)

        if len(rects) > 0:
            rect_id = np.argmax(scores)
            rect = rects[rect_id]
            # rect = rects[0]
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
        else:
            shape = None

        return rects, scores, idx, shape

    def process(self, img, face_data=None):
        img = np.array(img)
        orig_size = img.shape

        if np.max(img.shape[:2]) > 1024:
            if img.shape[0] < img.shape[1]:
                img_res = resize_pil(img, width=1024)
            else:
                img_res = resize_pil(img, height=1024)
        else:
            img_res = img

        resize_ratio = orig_size[0] / img_res.shape[0]
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        if face_data is None:
            rects, scores, idx = self.detector.run(gray, 1, -1)

            if len(rects) > 0:
                rect_id = np.argmax(scores)
                rect = rects[rect_id]
                # rect = rects[0]
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)
                # from here
                print()

        else:
            rects, scores, idx, shape = face_data

        loc = [0, 0]

        if len(rects) > 0:

            ## rect_id = np.argmax(scores)
            ## rect = rects[rect_id]
            ## # rect = rects[0]
            ## shape = self.predictor(gray, rect)
            ## shape = shape_to_np(shape)
            # (x, y, w, h) = rect_to_bb(rect)
            e0 = np.array(shape[38])
            e1 = np.array(shape[43])
            # e0 = np.array(shape[0])
            # e1 = np.array(shape[16])
            m0 = np.array(shape[48])
            m1 = np.array(shape[54])

            x_p = e1 - e0
            y_p = 0.5 * (e0 + e1) - 0.5 * (m0 + m1)
            c = 0.5 * (e0 + e1) - 0.1 * y_p
            s = np.max([4.0 * np.linalg.norm(x_p), 3.6 * np.linalg.norm(y_p)])
            xv = x_p - np.dot(self.R_90, y_p)
            xv /= np.linalg.norm(xv)
            yv = np.dot(self.R_90, y_p)

            s *= resize_ratio
            c[0] *= resize_ratio
            c[1] *= resize_ratio

            c1_ms = np.max([0, int(c[1] - s / 2)])
            c1_ps = np.min([img.shape[0], int(c[1] + s / 2)])
            c0_ms = np.max([0, int(c[0] - s / 2)])
            c0_ps = np.min([img.shape[1], int(c[0] + s / 2)])

            top = -np.min([0, int(c[1] - s / 2)])
            bottom = -np.min([0, img.shape[0] - int(c[1] + s / 2)])
            left = -np.min([0, int(c[0] - s / 2)])
            right = -np.min([0, img.shape[1] - int(c[0] + s / 2)])

            loc[0] = int(c1_ms)
            loc[1] = int(c0_ms)

            img = img[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]#here do something?

            if self.enable_segment:
                mask = self.seg_gen.segment(img)
            else:
                mask = None
            # mask = mask[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

            # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if self.enable_segment:
                mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

            border = [top, bottom, left, right]

            crop_sz = img.shape
            if np.max(img.shape[:2]) > 1024:
                img = cv2.resize(img, (1024, 1024))

                if self.enable_segment:
                    mask = cv2.resize(mask, (1024, 1024))
        else:
            img = None
            mask = None
            crop_sz = None
            border = None

        return img, mask, loc, crop_sz, border



    def process_side_offset(self, img, face_data=None):
        img = np.array(img)
        orig_size = img.shape

        if np.max(img.shape[:2]) > 1024:
            if img.shape[0] < img.shape[1]:
                img_res = resize_pil(img, width=1024)
            else:
                img_res = resize_pil(img, height=1024)
        else:
            img_res = img

        resize_ratio = orig_size[0] / img_res.shape[0]
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        if face_data is None:
            rects, scores, idx = self.detector.run(gray, 1, -1)

            if len(rects) > 0:
                rect_id = np.argmax(scores)
                rect = rects[rect_id]
                # rect = rects[0]
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)
                # from here
                print()


        else:
            rects, scores, idx, shape = face_data

        loc = [0, 0]

        if len(rects) > 0:

            box = compute_face_box(gray,shape)
            x_min = max(0,box.left())
            y_min = max(0,box.top())
            x_max = box.right()
            y_max = box.bottom()

            img = img[int(y_min*resize_ratio):int(y_max*resize_ratio), int(x_min*resize_ratio):int(x_max*resize_ratio)]#here do something?

            if self.enable_segment:
                mask = self.seg_gen.segment(img)
            else:
                mask = None
            # mask = mask[int(c1_ms):int(c1_ps), int(c0_ms):int(c0_ps)]

            # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # if self.enable_segment:
            #     mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
            #
            # border = [top, bottom, left, right]
            border = [0, 0, 0, 0]

            crop_sz = img.shape
            if np.max(img.shape[:2]) > 1024:
                img = cv2.resize(img, (1024, 1024))

                if self.enable_segment:
                    mask = cv2.resize(mask, (1024, 1024))
        else:
            img = None
            mask = None
            crop_sz = None
            border = None

        return img, mask, loc, crop_sz, border


def compute_face_box(frame, lm, border_fraction=0.1):
    x_min, y_min = np.min(lm, axis=0)
    x_max, y_max = np.max(lm, axis=0)

    width = x_max - x_min
    height = y_max - y_min

    x_min -= width *  border_fraction*1
    x_max += width *  border_fraction*1
    y_min -= height * border_fraction*6
    y_max += height * border_fraction

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    max_width = max(x_max - x_min, y_max - y_min)
    x_min = max(int(x_center - max_width / 2), 0)
    y_min = max(int(y_center - max_width / 2), 0)

    x_max = min(int(x_min + max_width), frame.shape[1])
    y_max = min(int(y_min + max_width), frame.shape[0])
    x_min = x_max - max_width
    y_min = y_max - max_width

    return dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))

# img_path_orig = '/home/tushar/FR/demo_FR/face_relight/1.png'
# import os
# test_path = '/home/tushar/FR/demo_FR/face_relight/test_data/fail'
#
# for im in os.listdir(test_path):
#     img_path_orig = os.path.join(test_path,im)
#
#     img_orig = cv2.imread(img_path_orig)
#     model_dir = osp.abspath('/home/tushar/FR/demo_FR/face_relight/demo/data/model/')
#     input_processor = InputProcessor(model_dir)
#     face_data = input_processor.get_face_data(img_orig)
#     rects, scores, idx, shape = face_data
#     # np.savez(face_data_id_path, rects=rects, scores=scores, idx=idx, shape=shape)
#     img_orig_proc = input_processor.process_2(img_orig, face_data)[0]
#     cv2.imwrite(im,img_orig_proc)