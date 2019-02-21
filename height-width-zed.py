import numpy as np
import cv2
import sys
import imutils
#video_path = 'M6 Motorway Traffic.mp4'
cv2.ocl.setUseOpenCL(False)

version = cv2.__version__.split('.')[0]

#read video file
cap = cv2.VideoCapture(0)


fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

print(cv2.__version__)
font = cv2.FONT_HERSHEY_SIMPLEX
while (cap.isOpened):

    #if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if ret==True:
        #apply background substraction
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = fgbg.apply(frame)
        #gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(fgmask, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        #thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print(len(cnts))
        if(len(cnts) > 0):
            c = max(cnts, key=cv2.contourArea)
            cn = max(c, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            hull = cv2.convexHull(c)
            cv2.drawContours(frame, hull, -1, (0, 255, 255), 2)
            cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
            cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
            cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
            cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

        cv2.imshow('foreground and background',gray)
        cv2.imshow('rgb',frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
