import math

import cv2
import numpy as np
import random as rd

NO_MOD = 0
FACE_GLITCH = 1
FACE_DRAG = 2
EYE_CENSOR = 3
EYE_DRAG = 4

EYES = 100
FACE = 101


def determine_face_mod(eyes_present):
    function_list = [
        (lambda x, y: x, FACE),
        (face_glitch, FACE),
        (face_drag, FACE),
        (eye_censor, EYES),
        (eye_drag, EYES)
    ]

    function_index = rd.randint(0, 4) if eyes_present else rd.randint(0, 2)

    return function_list[function_index]


def eye_drag(img, eyes):
    # make sure there are only two eyes per face
    if len(eyes) > 2:
        eye1 = eyes[0]
        eye2 = eyes[0]
        size = 0
        for itr in range(0, len(eyes)):
            if eyes[itr][2] * eyes[itr][3] > size:
                size = eyes[itr][2] * eyes[itr][3]
                eye1 = eyes[itr]
        size = 0
        for itr in range(0, len(eyes)):
            if eyes[itr][2] * eyes[itr][3] > size and not np.array_equal(eyes[itr], eye1):
                size = eyes[itr][2] * eyes[itr][3]
                eye2 = eyes[itr]
        eyes = [eye1, eye2]

    # there should only be two eyes now
    for eye in eyes:
        # find width of eye
        iwid = eye[2]
        strp = int(round(iwid / 20.))
        num_glitches = int(eye[2] / strp)
        line = rd.randint(1, eye[3])
        line += eye[1]
        line = int(eye[1] + eye[3] / 2)
        for itr in range(0, num_glitches):
            # edit the second parameter to change eye drop chance
            drop = rd.randint(10, 200)
            # if the line drop is too low, shorten it
            if line + drop > img.shape[0]:
                drop = img.shape[0] - line
            img[line:line + drop, eye[0] + itr * strp:eye[0] + itr * strp + strp] = \
                img[line, eye[0] + itr * strp:eye[0] + itr * strp + strp]


def eye_censor(img, eyes):
    if len(eyes) < 2:
        print("Failed to generate censor, less than two eyes present")
        return
    # cenH = 40
    # get centroids of eyes
    c1 = np.array([eyes[0][0] + eyes[0][2] / 2.0, eyes[0][1] + eyes[0][3] / 2.0])
    c2 = np.array([eyes[1][0] + eyes[1][2] / 2.0, eyes[1][1] + eyes[1][3] / 2.0])
    # find the corners of the bar
    # find vector of the two centroids
    vec = c1 - c2
    # unitize vector
    vec = vec / (vec[0] ** 2.0 + vec[1] ** 2.0) ** 0.5
    # perpendicular vector
    per_vec = np.array([vec[1], vec[0] * (-1)])
    # change these value to adjust height and width of
    # censor bar
    w_ex = 40
    mag = 75
    cr1 = per_vec * w_ex + c1
    cr2 = c1 - per_vec * w_ex
    cr3 = per_vec * w_ex + c2
    cr4 = c2 - per_vec * w_ex
    cr1 += vec * mag
    cr2 += vec * mag
    cr3 -= vec * mag
    cr4 -= vec * mag
    # round all values
    pts = np.array([cr1, cr2, cr4, cr3])
    cv2.fillPoly(img, np.array([pts], dtype=np.int32), (0, 0, 0))
    #########################################################
    # The following code is incomplete. It's purpose is to randomly
    # add text to the censor bar
    # roll to see if to add text
    # textc = rd.randint(0,2)
    # textc = 1
    # if textc == 1:
    #     text = open("elements/censor.txt","r")
    #     allText = text.read()
    #     possText = allText.split(";")
    #     dec = rd.randint(0,len(possText))
    #     use = possText[dec]
    #     #calculate text position and angle
    #     # info = posAndAngle(pts,use)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img,use,(int(cr1[0]),int(cr1[1])), font, 1,(255,255,255),2,cv2.LINE_AA)
    ############################################################


def face_drag(img, face):
    h, w, d = img.shape
    # 0 is horizontal 1 is veritical
    ornt = rd.randint(0, 2)
    if ornt == 0:
        line = rd.randint(face[1] + 25, face[1] + face[3] - 25)
        # 0 is up 1 is down
        direction = rd.randint(0, 2)
        if direction == 0:
            img[0:line, face[0]:face[0] + face[2]] = img[line, face[0]:face[0] + face[2]]
        else:
            img[line:h, face[0]:face[0] + face[2]] = img[line, face[0]:face[0] + face[2]]
    else:
        line = rd.randint(face[0] + 25, face[0] + face[2] - 25)
        # 0 is left 1 is right
        direction = rd.randint(0, 2)
        if direction == 0:
            img[face[1]:face[1] + face[3], 0:line] = img[face[1]:face[1] + face[3], line:line + 1]
        else:
            img[face[1]:face[1] + face[3], line:w] = img[face[1]:face[1] + face[3], line:line + 1]


def face_glitch(img, face):
    height, width, d = img.shape
    # pixels segments of 40
    div = rd.randint(10, 100)
    strp = int(round(face[3] / (div * 1.0)))
    num_glitches = face[3] / strp
    if type(num_glitches) == np.float64:
        num_glitches = math.floor(num_glitches)
    for itr in range(0, num_glitches):
        # play with the second parameter to increase "glitchiness"
        rng = rd.randint(15, 100)
        right_ext = face[0] + face[2] + rng
        left_ext = face[0] + face[2] - rng
        # make sure extremes don't go out of bounds
        if left_ext < 0:
            left_ext = 0
        if right_ext >= width:
            right_ext = width
        # randomize static direction
        # 1 moves left, 2 moves right
        dec = rd.randint(1, 2)
        back_bound = face[0] + rng
        if dec % 2 == 0:
            diff = 0
            # make corrections if glitch falls outside of image
            if face[0] + face[2] + rng >= width:
                diff = face[0] + face[2] + rng - width
            img[face[1] + (itr * strp):face[1] + (itr * strp + strp), (face[0] + rng):right_ext] = \
                img[face[1] + (itr * strp):face[1] + (itr * strp + strp), face[0]:face[0] + face[2] - diff]
        else:
            diff = 0
            # make corrections if glitch falls outside of image
            if back_bound < 0:
                diff = abs(back_bound)
                back_bound = 0
            old = img[face[1] + (itr * strp):face[1] + (itr * strp + strp), back_bound:left_ext]
            new = img[face[1] + (itr * strp):face[1] + (itr * strp + strp), face[0]:face[0] + face[2] - diff]
            if old.shape != new.shape:
                print("Shape mismatch: %s vs %s" % (old.shape, new.shape, ))
                return

            img[face[1] + (itr * strp):face[1] + (itr * strp + strp), back_bound:left_ext] = \
                img[face[1] + (itr * strp):face[1] + (itr * strp + strp), face[0]:face[0] + face[2] - diff]
