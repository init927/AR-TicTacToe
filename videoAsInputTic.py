import cv2
import imutils
import numpy as np
import math


cap = cv2.VideoCapture(0)
fps = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    fps+=1
    k = cv2.waitKey(1)

    if (k%256 == 32):
    	cv2.imwrite("photo.png",frame)
    	print("written!")
    	k=1

    if (fps>=25):
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    fps=0
	    clonei = gray.copy()
	    # r = 500.0 / gray.shape[1]
	    # dim = (500, int(gray.shape[0] * r))
	    # resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
	    # Display the resulting frame
	    # cv2.imshow('frame',resized)
	    gray = clonei.copy()
	    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	    wide = cv2.Canny(blurred, 10, 200,apertureSize = 3)
	    blurred = cv2.GaussianBlur(wide, (7, 7), 0)
	    (T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	    gray = cv2.bitwise_not(threshInv)
	    clonecircle = gray.copy()
	    cnts = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	    a=96
	    perimax=0
	    for (i, c) in enumerate(cnts):
	    	area = cv2.contourArea(c)
	    	perimeter = cv2.arcLength(c, True)
	    	if(perimax<perimeter):
	    		perimax=perimeter
	    		valuec=c
	    		M = cv2.moments(c)
	    		cX = int(M["m10"] / M["m00"])
	    		cY = int(M["m01"] / M["m00"])
	    cv2.drawContours(gray, [valuec], -1, (0), 20)
	    clone = gray.copy()
	    (x, y, w, h) = cv2.boundingRect(valuec)
	    cv2.rectangle(clonecircle, (x, y), (x + w, y + h), ( 255), 2)
	    cv2.line(clonecircle, (cX, cY), (cX + w, cY), 255)
	    cloneaftercircle = frame.copy()
	    cnts = cv2.findContours(clone.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	    a=96
	    for (i,c) in enumerate(cnts):
	    	area = cv2.contourArea(c)
	    	(x,y,w,h) = cv2.boundingRect(c)
	    	a=a+1;
	    	hull = cv2.convexHull(c)
	    	hullArea = cv2.contourArea(hull)
	    	if(hullArea==0):
	    		hullArea=1
	    	solidity = area / float(hullArea)
	    	M = cv2.moments(c)
	    	if(M["m00"]==0):
	    		M["m00"]=1
	    	Xp = int(M["m10"] / M["m00"])
	    	Yp = int(M["m01"] / M["m00"])
	    	cv2.line(clonecircle, (cX, cY), (cX + w, cY), 255)
	    	cXw=cX+w
	    	cv2.line(clonecircle, (cX, cY), (Xp, Yp), 255)
	    	p12 = math.sqrt(((cX-Xp)**2) + ((cY-Yp)**2))
	    	result = math.degrees(math.atan2(cY - cY, cXw - cX) - math.atan2(Yp - cY, Xp - cY))
	    	print("angle1 {} distance {}".format(result,p12))

	    	char = "?"
	    	if solidity > 0.9:
	    		char = "O"#+chr(a)
	    	elif solidity > 0.15:
	    		char = "X"#+chr(a)
	    	if char != "?":
	    		cv2.drawContours(frame,[c],-1,(0,255,0),3)
	    		cv2.putText(frame, char ,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,255,0),2)
	    	print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))


	    # r = 500.0 / frame.shape[1]
	    # dim = (500, int(frame.shape[0] * r))
	    # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()