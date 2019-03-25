# The following code was developed by Tim Chinenov
# The script turns a random image into a Vaporwave themed
# image. The program was written in opencv 3.3.1 and python 2.7

# To run the program call the following command in terminal
# python main.py

#To change the image being used, modify code on line 208
import cv2
import numpy as np
import random as rd
import os
from datetime import datetime

#The following function is used to determine the placement of
#text, at the moment it is incomplete
#function takes corners and text
# and resturns top left corner that
# centers text and the angle needed
def posAndAngle(pts,text):
    #find left top most coordinate
    dist = np.inf
    left = pts[0]
    for cr in pts:
        if (cr[0]**2+cr[1]**2)**0.5 < dist:
            dist = (cr[0]**2+cr[1]**2)**0.5
            left = cr
    #first find angle
    return 1;

def addElements(img):
    #get the number of elements
    allfiles = os.listdir("elements/black/")
    numFiles = len(allfiles)
    #get dimensions of main image
    imh,imw,imd = img.shape
    #randomize number of elements added
    numElements = rd.randint(2,4)
    #create a set to prevent element repetition
    usedels = set({})
    for num in range(numElements):
        file_name = "elements/black/ele_b"
        choice = rd.randint(1,numFiles)
        usedels.add(choice)
        #run again if element has been used already
        while choice not in usedels:
            choice = rd.randint(1,numFiles)

        file_name += str(choice) + ".png"
        element = cv2.imread(file_name,-1)
        if element is None:
            print(file_name+ " failed to load image")
            continue
        h,w,d = element.shape
        #adjust size if too big
        if h > imh*.5 or w > imw*.5:
            element = cv2.resize(element,(int(.5*w),int(.5*h)))
            h,w,d = element.shape
            #refuse to use this image, if this failed
            if h > imh or w > imw:
                print("Element too big, moving on")
                continue

        #get x coord and y coord on the image
        xpos = rd.randint(1,imw-w-1)
        ypos = rd.randint(1,imh-h-1)
        #make alpha channel
        alpha_s = element[:,:,2]/255.0
        alpha_1 = 1.0 - alpha_s
        for c in range(0,3):
            img[ypos:ypos+h,xpos:xpos+w,c] = (alpha_s*element[:,:,c]+alpha_1*img[ypos:ypos+h,xpos:xpos+w,c])

def faceGlitch(img,face):
    #pixels segments of 40
    div = rd.randint(10,100)
    strp = int(round(face[3]/(div*1.0)))
    numGlitches = face[3]/strp
    for itr in range(0,numGlitches):
        rng = rd.randint(15,100)
        rightExt = face[0]+face[2]+rng
        leftExt = face[0]+face[2]-rng
        #make sure extremes don't go out of bounds
        if leftExt < 0:
            leftExt = 0
        if rightExt >= width:
            rightExt = width-20
        #randomize static direction
        #1 moves left, 2 moves right
        dec = rd.randint(1,2)
        if dec%2 == 0:
            img[face[1]+(itr*strp):face[1]+(itr*strp+strp),(face[0]+rng):rightExt] = img[face[1]+(itr*strp):face[1]+(itr*strp+strp),face[0]:face[0]+face[2]]
        else:
            backBound = face[0]-rng
            diff = 0
            if backBound < 0:
                diff = abs(backBound)
                backBound = 0
            img[face[1]+(itr*strp):face[1]+(itr*strp+strp),(backBound):leftExt] = img[face[1]+(itr*strp):face[1]+(itr*strp+strp),face[0]:face[0]+face[2]-diff]

def faceDrag(img,face):
    h,w,d = img.shape
    #0 is horizontal 1 is veritical
    ornt = rd.randint(0,2)
    if ornt == 0:
        line = rd.randint(face[1]+25,face[1]+face[3]-25)
        #0 is up 1 is down
        dir = rd.randint(0,2)
        if dir == 0:
            img[0:line,face[0]:face[0]+face[2]] = img[line,face[0]:face[0]+face[2]]
        else:
            img[line:h,face[0]:face[0]+face[2]] = img[line,face[0]:face[0]+face[2]]
    else:
        line = rd.randint(face[0]+25,face[0]+face[2]-25)
        #0 is left 1 is right
        dir = rd.randint(0,2)
        if dir == 0:
            img[face[1]:face[1]+face[3],0:line] = img[face[1]:face[1]+face[3],line:line+1]
        else:
            img[face[1]:face[1]+face[3],line:w] = img[face[1]:face[1]+face[3],line:line+1]

def eyeCensor(img,eyes):
    if len(eyes) < 2:
        print("Failed to generate censor, less than two eyes present")
        return
    cenH = 40
    #get centroids of eyes
    c1 = np.array([eyes[0][0] + eyes[0][2]/2.0,eyes[0][1] + eyes[0][3]/2.0])
    c2 = np.array([eyes[1][0] + eyes[1][2]/2.0,eyes[1][1] + eyes[1][3]/2.0])
    #find the corners of the bar
    #find vector of the two centroids
    vec = c1-c2
    #unitize vector
    vec = vec/(vec[0]**2.0+vec[1]**2.0)**0.5
    #perpendicular vector
    perVec = np.array([vec[1],vec[0]*(-1)])
    #change these value to adjust height and width of
    #censor bar
    wEx = 40
    mag = 75
    cr1 = perVec*wEx+c1
    cr2 = c1 - perVec*wEx
    cr3 = perVec*wEx+c2
    cr4 = c2 - perVec*wEx
    cr1 += vec*mag
    cr2 += vec*mag
    cr3 -= vec*mag
    cr4 -= vec*mag
    #round all values
    pts = np.array([cr1,cr2,cr4,cr3])
    cv2.fillPoly(img,np.array([pts],dtype=np.int32),(0,0,0))
    #########################################################
    #The following code is incomplete. It's purpose is to randomly
    #add text to the censor bar
    #roll to see if to add text
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

def eyeDrag(img,eyes):
    #make sure there are only two eyes per face
    if len(eyes) > 2:
        eye1 = eyes[0]
        eye2 = eyes[0]
        size = 0
        for itr in range(0,len(eyes)):
            if eyes[itr][2]*eyes[itr][3] > size:
                size = eyes[itr][2]*eyes[itr][3]
                eye1 = eyes[itr]
        size = 0
        for itr in range(0,len(eyes)):
            if eyes[itr][2]*eyes[itr][3] > size and not np.array_equal(eyes[itr],eye1):
                size = eyes[itr][2]*eyes[itr][3]
                eye2 = eyes[itr]
        eyes = [eye1,eye2]
    #there should only be two eyes now
    for eye in eyes:
        #find width of eye
        iwid = eye[2]
        strp = int(round(iwid/20.))
        numGlitches = eye[2]/strp
        line = rd.randint(1,eye[3])
        line += eye[1]
        line = eye[1] + eye[3]/2
        for itr in range(0,numGlitches):
            drop = rd.randint(10,200)
            img[line:line+drop,eye[0]+itr*strp:eye[0]+itr*strp+strp] = img[line,eye[0]+itr*strp:eye[0]+itr*strp+strp]


if __name__ == "__main__":
    #seed the random generator
    rd.seed(datetime.now())
    #load files for facial and eye cascade
    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

    #load main image from local file
    img = cv2.imread("testImgs/testface9.png")
    height,width,depth = img.shape
    #turn image gray for detecting face and eyes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find all the faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #go through each face
    for face in faces:
        y = face[1]
        x = face [0]
        w = face [2]
        h = face [3]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #find each eye. Modify second and third parameter if
        #feature detection is poor
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
        for eye in eyes:
            eye[0] += face[0]
            eye[1] += face[1]

        #randomize which face modification will be performed
        faceMod = rd.randint(1,4)
        if faceMod >= 4 and len(eyes) == 0:
            faceMod = rd.randint(0,4)

        #0 - no mod
        #1 - face glitch
        #2 - face drag
        #3 - eye drag
        #4 - eye sensor
        if faceMod == 1:
            faceGlitch(img,face)
        elif faceMod == 2:
            faceDrag(img,face)
        elif faceMod == 3:
            eyeDrag(img,eyes)
        elif faceMod == 4:
            eyeCensor(img,eyes)

    # Add elements to image
    addElements(img)
    #if there are no faces, just add more elements!
    if len(faces) < 1:
        addElements(img)

    #randomize if high contrast is used
    choice = rd.randint(0,1)
    if choice == 1:
        #edit alpha and beta to adjust contrast levels
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=35)

    #randomize if high noise is used
    choice = rd.randint(0,1)
    if choice == 1:
        row,col,ch= img.shape
        mean = 0
        #edit var to modify the amount of noise in the image
        var = 15
        sigma = var**1
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = (img + gauss)
        cv2.normalize(noisy,  noisy, 0, 1, cv2.NORM_MINMAX)
        img = noisy

    #The following code is useful to determine if faces and eyes
    #are being read correctly. Uncommenting will draw boxes around
    #found features.
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
    #     #edit the second and third parameter if feature detection is poor
    #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.namedWindow("pic",cv2.WINDOW_NORMAL)
    cv2.imshow("pic",img)
    cv2.waitKey(0)
