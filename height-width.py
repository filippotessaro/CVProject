import numpy as np
import cv2
import sys

#video_path = 'M6 Motorway Traffic.mp4'
cv2.ocl.setUseOpenCL(False)

version = cv2.__version__.split('.')[0]

#read video file
cap = cv2.VideoCapture(0)

#check opencv version
if version == '2' :
	print(cv2.__version__)
	fgbg = cv2.BackgroundSubtractorMOG2()
if version >= '3':
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
		#fgmask = fgbg.apply(frame)
		#(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		#grey  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		blur =cv2.GaussianBlur(fgmask,(5,5),0)
		ret,th = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)
		dilated = cv2.dilate(th,np.ones((3,3),np.uint8),iterations=3)
		(im2, contours, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		#looping for contours
		for c in contours:
			if cv2.contourArea(c) < 1000:
				continue

			#get bounding box from countour
			(x, y, w, h) = cv2.boundingRect(c)
			width = 'width=' + str(w)
			height = 'height=' + str(h) + width
			#draw bounding box & text of measures
			cv2.putText(frame, height ,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow('foreground and background',dilated)
		cv2.imshow('rgb',frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


cap.release()
cv2.destroyAllWindows()
