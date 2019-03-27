# The following code was developed by Tim Chinenov
# The script turns a random image into a Vaporwave themed
# image. The program was written in opencv 3.3.1 and python 2.7

# To run the program call the following command in terminal
# python main.py

import sys
import logging

import cv2

from vaporwave import vaporize

ESCAPE_KEY = 27

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main():

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


if __name__ == "__main__":
    main()
