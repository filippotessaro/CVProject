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
import pyautogui
from scipy.spatial import distance
import statistics as s
import pandas as pd

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

person_name = 'measures/'+ str(sys.argv[1]) + '-'
kg_IN = float(sys.argv[2])

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
camera_settings.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720  # Use HD720 video mode (default fps: 60)

str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

#----Added parameters
threshold = 60  # BINARY threshold
blurValue = 15  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

area_pixel = 0
area_msquares = 0

# variables
isBgCaptured = 0  # bool, whether the background captured

def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


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
    duBoisWeight = (body_surface / (0.20247 * (height**0.725) )) ** (1/0.425)
    #print('mostellerWeight:', mostellerWeight, 'duBoisWeight:', duBoisWeight)

    weights.append(mostellerWeight)
    weights.append(duBoisWeight)
    return weights


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
    print("  Reset all parameters:               a")
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
    elif key == ord('a'):  # for 'a' key
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


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def printAlpha(alpha):
    print('Alpha:', str(alpha))

def printBeta(beta):
    print('Beta:', str(beta))

def applyAlpha(alpha_percentage):
    return alpha_percentage * ((3.0 - 1.0)/100) + 1.0

#-------------------------------------MAIN------------------------------------------------
print("Running...")
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA # Use ULTRA depth mode
init.coordinate_units = sl.UNIT.UNIT_MILLIMETER # Use millimeter units (for depth measurements)
font = cv2.FONT_HERSHEY_SIMPLEX

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
point_cloud = sl.Mat()

gamma = 1
alpha = 1.0 # Simple contrast control
beta = 0    # Simple brightness control
alpha_percentage = 0

cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
cv2.createTrackbar('alpha', 'trackbar', alpha_percentage, 100, printAlpha)
cv2.createTrackbar('beta', 'trackbar', beta, 100, printBeta)

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
isBgCaptured = False

#initialized dataframe for measures
df = pd.DataFrame(columns=['height', 'width', 'shapeArea', 'dubois', 'mosteller', 'avgError'])

print_camera_information(cam)
print_help()

#f = open(person_name + "measures.txt", "a")
#h_file = open(person_name + "height.txt", "a")

key = ''
while True:  # for 'q' key
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)

        cam.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)

        frame = mat.get_data()
        frame  = cv2.cvtColor(frame,cv2.COLOR_RGBA2RGB)

        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        alpha_percentage = cv2.getTrackbarPos('alpha', 'trackbar')
        alpha = applyAlpha(alpha_percentage)

        beta = cv2.getTrackbarPos('beta', 'trackbar')
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.imshow('trackbar', frame)

        settings(key, cam, runtime, mat)

        #  Main operation
        if isBgCaptured:  # this part wont run until background captured
            blur = cv2.GaussianBlur(frame,(5,5),0)

            img =  bgModel.apply(blur, learningRate=learningRate)
            cv2.imshow('mask', img)

            #Threshold phase
            ret,thresh = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
            cv2.imshow('threshold', thresh)

            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)

            if length > 0:
                c = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(c)
                area_pixel = w * h

                #Get the 3D coordinates of the points
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extLeft3D = point_cloud.get_value(extLeft[0], extLeft[1])

                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extRight3D = point_cloud.get_value(extRight[0], extRight[1])

                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extTop3D = point_cloud.get_value(extTop[0], extTop[1])

                extBot = tuple(c[c[:, :, 1].argmax()][0])
                extBot3D = point_cloud.get_value(extBot[0], extBot[1])

                #width an height of the extreme points
                #real_height = math.sqrt((extTop3D[1][0]-extBot3D[1][0])**2 + (extTop3D[1][1]-extBot3D[1][1])**2 + (extTop3D[1][2]-extBot3D[1][2])**2)
                #real_width = math.sqrt((extLeft3D[1][0]-extRight3D[1][0])**2 + (extLeft3D[1][1]-extRight3D[1][1])**2 + (extLeft3D[1][2]-extRight3D[1][2])**2)

                real_height = math.sqrt((extTop3D[1][0]-extBot3D[1][0])**2 + (extTop3D[1][1]-extBot3D[1][1])**2)
                real_width = math.sqrt((extLeft3D[1][0]-extRight3D[1][0])**2 + (extLeft3D[1][1]-extRight3D[1][1])**2)
                print('Height', real_height)
                #estimate real height and width in mm by euclidean distance
                if(not np.isnan(real_height) and not np.isinf(real_height) and not np.isnan(real_width) and not np.isinf(real_width)):
                    #find FRONT human surface area in m2 by proportion
                    #h_file.write("{0:.2f}".format(real_height) + '\n')
                    if (real_height < 2000 and (not real_height == 0) and real_width < 1250 and (not real_width == 0)):
                        area_msquares = (real_height/1000) * (real_width/1000)
                        shape_real_m2 = (cv2.contourArea(c) * (area_msquares/area_pixel) + 0.15) *2

                        #weight estimation, double the surface area for the back
                        measures = weightEstimation(shape_real_m2, real_height/1000)
                        avgWeight = s.mean(measures)

                        error = (abs(kg_IN-avgWeight)/kg_IN) * 100

                        if (avgWeight < 150 and avgWeight > 40):
                            #f.write("{0:.2f}".format(avgWeight) + '\n')
                            df = df.append(pd.Series([real_height, real_width, shape_real_m2, measures[0], measures[1], error], index=df.columns ), ignore_index=True)
                            cv2.putText(frame, "Weight: " + "{0:.2f}".format(avgWeight) + 'Kg' ,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(frame, "Height: " + "{0:.2f}".format(real_height) + 'mm' ,(50,100), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(frame, "Width: " + "{0:.2f}".format(real_width) + 'mm' ,(50,140), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(frame, "Surface: " + "{0:.2f}".format(shape_real_m2) + 'm2' ,(50,180), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.putText(frame, "ERROR: " + "{0:.2f}".format(error) + ' %' ,(50,220), font, 1,(0,0,255),2,cv2.LINE_AA)


                hull = cv2.convexHull(c)
                hull = cv2.convexHull(c)

                #draw all the shapes
                cv2.drawContours(frame, [c], 0, (0, 255, 0), 2)
                cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)
                cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
                cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
                cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
                cv2.circle(frame, extBot, 8, (255, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('output', frame)


        key = cv2.waitKey(10)
        if key == 27:  # press ESC to exit
            break
        elif key == ord('b'):  # press 'b' to capture the background
            #Background model initialization
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = True
            print('!!!Background Captured!!!')
        elif key == ord('r'):  # press 'r' to reset the background
            bgModel = None
            isBgCaptured = 0
            print('!!!Reset BackGround!!!')

cv2.destroyAllWindows()
#f.close()
#h_file.close()
cam.close()
#export csv
df.to_csv(person_name + 'measuresdataframe.csv', sep='\t')
print("\nFINISH")
