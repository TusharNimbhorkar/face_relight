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
predictor = dlib.shape_predictor('/home/nedko/storage/huawei/models/shape_predictor_68_face_landmarks.dat')



# obj_dir = '/home/tushar/data2/face_rel_multipie'
obj_dir = '/home/nedko/face_relight/test_data/portrait_test'
objs = sorted(os.listdir(obj_dir))

count = len(objs)
save_folder = '/home/nedko/face_relight/outputs/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for o in objs:
    save_folder_obj = os.path.join(save_folder,o)
    if not os.path.exists(save_folder_obj):
        os.makedirs(save_folder_obj)

    person_dir = os.path.join(obj_dir,o)
    #
    # save_folder_obj = save_folder
    # person_dir = obj_dir

    if os.path.exists(o+'_07.png'):
        center_frame = os.path.join(person_dir,o+'_07.png')
    else:
        center_frame = os.path.join(person_dir, os.listdir(person_dir)[0])

    print(count, center_frame)
    count = count-1
    # continue

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(center_frame)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    ilums = os.listdir(person_dir)

    def R(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    R_90 = R(np.deg2rad(90))

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)


        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        e0 = np.array(shape[38])
        e1 = np.array(shape[43])
        m0 = np.array(shape[48])
        m1 = np.array(shape[54])

        x_p = e1-e0
        y_p = 0.5*(e0+e1) - 0.5*(m0+m1)
        c = 0.5*(e0+e1) - 0.1*y_p
        s = np.max([4.0*np.linalg.norm(x_p),3.6*np.linalg.norm(y_p)])
        xv = x_p - np.dot(R_90,y_p)
        xv /= np.linalg.norm(xv)
        yv = np.dot(R_90,y_p)
        print('Center: ', c, 'Size: ', s)

        t_x = 0
        t_y = 0
        if c[0]-s/2 < 0:
            t_x = -c[0]+s/2
        if c[1]-s/2 < 0:
            t_y = -c[1]+s/2

        c[0] += t_x
        c[1] += t_y
        if t_x != 0 or t_y != 0:
            image = imutils.translate(image, t_x, t_y)

        # image = image[:,:, ::-1]
        # sz = 3
        # to_draw = [e0,e1,m0,m1]
        # # print(image.shape)
        # for point in to_draw:
        #     image[point[1]-sz:point[1]+sz, point[0]-sz:point[0]+sz, :] = [255,0,0]
        #
        # print(image.shape)
        # image = np.ascontiguousarray(image)

        # cv2.rectangle(image, (int(c[0]-s/2), int(c[1]-s/2)), (int(c[0] + s/2), int(c[1] + s/2)), (0, 255, 0), 2)

        # plt.imshow(image)
        # plt.show()
        # print(int(c[1] - s/2),int(c[1] + s/2), int(c[0] - s/2),int(c[0]+s/2))
        # plt.imshow(image[int(c[1] - s/2):int(c[1] + s/2), int(c[0] - s/2):int(c[0]+s/2)])
        # plt.show()

        for il in ilums:
            im1 = cv2.imread(os.path.join(person_dir,il))
            save_file_name = os.path.join(save_folder_obj,il)
            cv2.imwrite(save_file_name,im1[int(c[1] - s/2):int(c[1] + s/2), int(c[0] - s/2):int(c[0]+s/2)])