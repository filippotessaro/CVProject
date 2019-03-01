import app as app
import cv2
import numpy as np
import copy
import math
import time
from pynput.mouse import Button, Controller
import pyautogui

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def weightEstimation(body_surface, height):
    '''
    given the body surface in m^2 and the height in m
    returns a list of estimated weights in kg
    '''
    weights = []
    #Mosteller k, body_surface in m^2, height cm
    cm_height = height * 100
    mostellerWeight = (3600 * body_surface **2)/cm_height
    #DuBois & DuBois height m
    duBoisWeight = (surface / (0.20247 * (height**0.725) )) ** (1/0.425)
    weights.extend(mostellerWeight, duBoisWeight)
    return weights

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

#bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
isBgCaptured = 0

while camera.isOpened():
    # detect people in the image
    ret, frame = camera.read()
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow('output', frame)






    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
print('!!!Trigger On!!!')
