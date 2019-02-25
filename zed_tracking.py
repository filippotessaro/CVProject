import cv2
import pyzed.sl as sl

import numpy as np
import sys

import app as app
import cv2
import numpy as np
import copy
import math
import time
from pynput.mouse import Button, Controller
import pyautogui


camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

#----Added parameters
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
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res
#-----------------------------------------

def main():
    print("Running...")
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA # Use ULTRA depth mode
    init.coordinate_units = sl.UNIT.UNIT_MILLIMETER # Use millimeter units (for depth measurements)


    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    depth_map = sl.Mat()

    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    isBgCaptured = 0

    print_camera_information(cam)
    print_help()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            #cam.retrieve_measure(depth_map, sl.MEASURE.MEASURE_DEPTH) # Retrieve depth

            point_cloud = sl.Mat()
            cam.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)

            frame = mat.get_data()
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
            frame = cv2.flip(frame, 1)  # flip the frame horizontally
            #cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                          #(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
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
                    hull = cv2.convexHull(c)
                    drawing = np.zeros(img.shape, np.uint8)
                    cv2.drawContours(drawing, [c], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                    cv2.circle(drawing, extLeft, 8, (0, 0, 255), -1)
                    cv2.circle(drawing, extRight, 8, (0, 255, 0), -1)
                    cv2.circle(drawing, extTop, 8, (255, 0, 0), -1)
                    cv2.circle(drawing, extBot, 8, (255, 255, 0), -1)

                cv2.imshow('output', drawing)


            #key = cv2.waitKey(5)
            #settings(key, cam, runtime, mat)
            # Keyboard OP
            key = cv2.waitKey(10)
            if key == 27:  # press ESC to exit
                break
            elif key == ord('b'):  # press 'b' to capture the background
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
                isBgCaptured = 1
                print('!!!Background Captured!!!')
            elif key == ord('r'):  # press 'r' to reset the background
                bgModel = None
                triggerSwitch = False
                isBgCaptured = 0
                print('!!!Reset BackGround!!!')
            elif key == ord('n'):
                triggerSwitch = True
        print('!!!Trigger On!!!')
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 115:  # for 's' key
        switch_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE, -1, True)
        print("Camera settings: reset")
    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)


def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")


def record(cam, runtime, mat):
    vid = sl.ERROR_CODE.ERROR_CODE_FAILURE
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:
        filepath = input("Enter filepath name: ")
        vid = cam.enable_recording(filepath)
        print(repr(vid))
        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:  # for spacebar
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
                    cam.record()
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
