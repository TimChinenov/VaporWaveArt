# The following code was developed by Tim Chinenov
# The script turns a random image into a Vaporwave themed
# image. The program was written in opencv 3.3.1 and python 2.7

# To run the program call the following command in terminal
# python main.py

# To change the image being used, modify code on line 208
import sys

import cv2
import logging
import numpy as np
import random as rd
import os
from functools import partial

import mods
from datetime import datetime

ESCAPE_KEY = 27

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# The following function is used to determine the placement of
# text, at the moment it is incomplete
# function takes corners and text
# and resturns top left corner that
# centers text and the angle needed
def pos_and_angle(pts):
    # find left top most coordinate
    dist = np.inf
    # left = pts[0]
    for cr in pts:
        if (cr[0] ** 2 + cr[1] ** 2) ** 0.5 < dist:
            dist = (cr[0] ** 2 + cr[1] ** 2) ** 0.5
            # left = cr
    # first find angle
    return 1


def add_elements(img):
    min_elements = 2
    max_elements = 4

    base_dir = "elements/black/"

    all_files = os.listdir(base_dir)
    rd.shuffle(all_files)

    # randomize number of elements added
    num_elements = rd.randint(min_elements, max_elements)
    # create a set to prevent element repetition
    added_counter = 0

    logger.info("Adding %d elements" % (num_elements, ))

    for file_name in map(partial(os.path.join, base_dir), all_files):
        if added_counter == num_elements:
            return

        success = add_single_element(img, file_name)
        if success:
            added_counter += 1


def add_single_element(img, file_name):
    imh, imw, imd = img.shape
    element = cv2.imread(file_name, -1)
    if element is None:
        logger.warning("Could not read file %s" % (file_name,))
        return False

    original_height, original_width, original_depth = element.shape
    # adjust size if too big
    if original_height > imh * .5 or original_width > imw * .5:
        element = cv2.resize(element, (int(.5 * original_width), int(.5 * original_height)))

        resized_height, resized_width, _ = element.shape
        # refuse to use this image, if this failed
        if resized_height > imh or resized_width > imw:
            logger.warning("Element %s too big, moving on" % (file_name,))
            return False
        # get x coord and y coord on the image
        from_x_pos = rd.randint(1, imw - resized_width - 1)
        from_y_pos = rd.randint(1, imh - resized_height - 1)
        # make alpha channel
        alpha_s = element[:, :, 2] / 255.0
        alpha_1 = 1.0 - alpha_s
        for c in range(0, 3):
            to_y_pos = from_y_pos + resized_height
            to_x_pos = from_x_pos + resized_width

            with_alpha_s = alpha_s * element[:, :, c]
            with_alpha_1 = alpha_1 * img[from_y_pos:to_y_pos, from_x_pos:to_x_pos, c]

            img[from_y_pos:to_y_pos, from_x_pos:to_x_pos, c] = with_alpha_s + with_alpha_1
    return True


def main():
    # seed the random generator
    rd.seed(datetime.now())
    # load files for facial and eye cascade

    img = vaporize()

    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    cv2.imshow("pic", img)

    while cv2.getWindowProperty("pic", cv2.WND_PROP_VISIBLE):
        key_code = cv2.waitKey(100)

        if key_code == ESCAPE_KEY:
            break
        elif key_code != -1:
            import time
            start = time.time()
            img = vaporize()
            cv2.imshow("pic", img)
            end = time.time()
            logger.info("Vaporizing and rendering took: %f seconds" % (end-start,))
    cv2.destroyAllWindows()
    sys.exit()


def vaporize(image_path="testImgs/testface9.png"):
    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

    # load main image from local file
    img = cv2.imread(image_path)
    # height, width, depth = img.shape

    # turn image gray for detecting face and eyes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find all the faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # go through each face
    for face in faces:
        y = face[1]
        x = face[0]
        w = face[2]
        h = face[3]
        roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # find each eye. Modify second and third parameter if
        # feature detection is poor
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
        for eye in eyes:
            eye[0] += face[0]
            eye[1] += face[1]

        # randomize which face modification will be performed
        eyes_present = len(eyes) >= 2
        mod_function, operates_on = mods.determine_face_mod(eyes_present)
        if operates_on == mods.EYES:
            mod_function(img, eyes)
        elif operates_on == mods.FACE:
            mod_function(img, face)

    # Add elements to image
    add_elements(img)
    # if there are no faces, just add more elements!
    if len(faces) < 1:
        add_elements(img)

    # randomize if high contrast is used
    choice = rd.randint(0, 1)
    if choice == 1:
        # edit alpha and beta to adjust contrast levels
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=35)

    # randomize if high noise is used
    choice = rd.randint(0, 1)
    if choice == 1:
        row, col, ch = img.shape
        mean = 0
        # edit var to modify the amount of noise in the image
        var = 15
        sigma = var ** 1
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = (img + gauss)
        cv2.normalize(noisy, noisy, 0, 1, cv2.NORM_MINMAX)
        img = noisy

    # The following code is useful to determine if faces and eyes
    # are being read correctly. Uncommenting will draw boxes around
    # found features.
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
    #     #edit the second and third parameter if feature detection is poor
    #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img


if __name__ == "__main__":
    main()
