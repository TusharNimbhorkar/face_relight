from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2,os
# import the necessary packages
from collections import OrderedDict
import matplotlib.pyplot as plt


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/tushar/face_relight/shape_predictor_68_face_landmarks.dat')



obj_dir = '/home/tushar/face_relight/multipie'
objs = sorted(os.listdir(obj_dir))

count = len(objs)
save_folder = '/home/tushar/face_relight/face_crops'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


for o in objs:
    print(count)
    count = count-1
    save_folder_obj = os.path.join(save_folder,o)
    if not os.path.exists(save_folder_obj):
        os.makedirs(save_folder_obj)

    person_dir = os.path.join(obj_dir,o)
    center_frame = os.path.join(person_dir,o+'_07.png')

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(center_frame)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    ilums = os.listdir(person_dir)


    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x-12, y-25), (x + w+12, y + h+25), (0, 255, 0), 2)
        plt.imshow(image[y - 25:y + h + 25, x - 12:x + w + 20])

        for il in ilums:
            im1 = cv2.imread(os.path.join(person_dir,il))
            save_file_name = os.path.join(save_folder_obj,il)
            cv2.imwrite(save_file_name,im1[y - 25:y + h + 25, x - 12:x + w + 20])