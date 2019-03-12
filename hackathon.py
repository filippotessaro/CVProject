import app as app
import cv2
import numpy as np
import copy
import math
import time
from pynput.mouse import Button, Controller
import pyautogui
import pandas as pd


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
globalCounterNext = 0
globalCounterPrevious = 0
mouse = Controller()


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

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

df = pd.DataFrame(columns=['height', 'width', 'shapeArea', 'weight'])


gamma = 1
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
cv2.createTrackbar('trh2', 'trackbar', gamma, 100, printThreshold)

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
isBgCaptured = 0

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    gamma = cv2.getTrackbarPos('trh2', 'trackbar')
    gamma = gamma if gamma > 0 else 0.1
    frame = adjust_gamma(frame, gamma=gamma)
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            c = max(contours, key=cv2.contourArea)
            #cn = max(c, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            hull = cv2.convexHull(c)
            drawing = np.zeros(img.shape, np.uint8)

            cv2.drawContours(drawing, [c], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            cv2.circle(drawing, extLeft, 8, (0, 0, 255), -1)
            cv2.circle(drawing, extRight, 8, (0, 255, 0), -1)
            cv2.circle(drawing, extTop, 8, (255, 0, 0), -1)
            cv2.circle(drawing, extBot, 8, (255, 255, 0), -1)
            (x, y, w, h) = cv2.boundingRect(c)

            ## TODO: height and width of the rectangule box
            ## TODO: proportion rectArea_REAL : rectAREA_PIXEL = shape_REAL : shape_PIXEL(OBTAINED FROM CONTOUR AREA)

            cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('output', drawing)


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
