import random as rd

import cv2
import numpy as np

from .elements import add_single_element, add_elements
from . import mods


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
