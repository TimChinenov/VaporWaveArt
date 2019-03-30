import logging
import os
import random as rd
from functools import partial

import cv2

logger = logging.getLogger("elements")


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
