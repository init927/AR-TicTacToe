import cv2
import imutils
import numpy as np
import math

bpos=cpos=dpos=epos=fpos=gpos=hpos=ipos=apos=10

img = cv2.imread('b.jpg')
# p1 is mid point
# double result = atan2(P3.y - P1.y, P3.x - P1.x) -
#                 atan2(P2.y - P1.y, P2.x - P1.x);

# Convert the img to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# Apply edge detection method on the image
edges = cv2.Canny(gray,50,150,apertureSize = 3)
 
# This returns an array of r and theta values
lines = cv2.HoughLines(edges,1,np.pi/180, 200)
 
# The below for loop runs till r and theta values 
# are in the range of the 2d array
for r,theta in lines[0]:
    print("theta:{}".format(theta))
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
    print("a:{}".format( math.degrees(a)))
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    print("b:{}".format(b))
    # x0 stores the value rcos(theta)
    x0 = a*r
    print("x0:{}".format(x0))
    # y0 stores the value rsin(theta)
    y0 = b*r
    print("y0:{}".format(y0))
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
    print("x1:{}".format(x1))
     
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))
    print("y1:{}".format(y1))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
    print("x2:{}".format(x2))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))
    print("y2:{}".format(y2))
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be 
    #drawn. In this case, it is red. 
    # cv2.line(img,(x1,y1), (x2,y2), (0,255,255),0)
    # cv2.circle(img,  (x1,y1), 1300, (0, 0, 255), -1)
    (x1,x2)= (x2,x1)
    (y1,y2)=  (y2,y1)
    x3 = x1+100
    y3 = y1
    # cv2.line(img,(x1,y1), (x3,y3), (0,255,255),0)
# p1 is mid point
# double result = atan2(P3.y - P1.y, P3.x - P1.x) -
# atan2(P2.y - P1.y, P2.x - P1.x);
    result = math.degrees(math.atan2(y3 - y1, x3 - x1) - math.atan2(y2 - y1, x2 - x1))
    # degreeofrotation= result - 90
    # print("result {}".format(result))
    if result<0:
        result=360+result
    if result>90 and result<170:
        degreeofrotation= result - 90
        rotated = imutils.rotate(img, -degreeofrotation)
        cv2.imshow("Rotated by 180 Degrees", rotated)
        cv2.waitKey(0)
    elif result>=170 and result<190:
        degreeofrotation= abs(result - 180)
        rotated = imutils.rotate(img, -degreeofrotation)
        cv2.imshow("Rotated by 180 Degreesh", rotated)
        cv2.waitKey(0)
    print("result {}".format(result))
# All the changes made in the input image are finally
# written on a new image houghlines.jpg
cv2.imwrite('houghlines3.jpg', rotated)

image = cv2.imread('houghlines3.jpg')
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
clonei = gray.copy()
cv2.imshow("Clone",clonei)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200,apertureSize = 3)
tight = cv2.Canny(blurred, 225, 250)
auto = imutils.auto_canny(blurred)
cv2.imshow("wide canny edge",wide)
blurred = cv2.GaussianBlur(wide, (7, 7), 0)
cv2.imshow("blured", blurred)
 
# apply Otsu's automatic thresholding -- Otsu's method automatically
# determines the best threshold value `T` for us
(T, threshInv) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cv2.imshow("Threshold", threshInv)
# print("Otsu's thresholding value: {}".format(T))
gray = cv2.bitwise_not(threshInv)
cv2.imshow("Threshold",gray)

clonecircle = gray.copy()
cnts = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
a=96

# for c in cnts:
# 	# compute the moments of the contour which can be used to compute the
# 	# centroid or "center of mass" of the region
# 	M = cv2.moments(c)
# 	cX = int(M["m10"] / M["m00"])
# 	cY = int(M["m01"] / M["m00"])
 
# # 	# draw the center of the contour on the image
# 	cv2.circle(clonecircle, (cX, cY), 10, (0, 255, 0), -1)
# 	((x, y), radius) = cv2.minEnclosingCircle(c)
# 	print("x{}y{}".format(x,y))
# 	cv2.circle(clonecircle, (int(x), int(y)), int(radius), ( 255), 2)
#  	cv2.imshow("Centroids", clonecircle)
# 	cv2.waitKey(0)

# for c in cnts:
# 	# fit a rotated bounding box to the contour and draw a rotated bounding box
# 	box = cv2.minAreaRect(c)
# 	box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
# 	cv2.drawContours(clonecircle, [box], -1, (255), 2)
 
# # show the output image
# cv2.imshow("Rotated Bounding Boxes", clonecircle)
# cv2.waitKey(0)
# cloneaftercircle = image.copy()

# clone the original image
 
# loop over the contours

# # show the output image
# cv2.imshow("Centroids", clone)
# cv2.waitKey(0) 


# for c in cnts:
# 	# fit a minimum enclosing circle to the contour
# 	((x, y), radius) = cv2.minEnclosingCircle(c)
# 	cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)
 
# # show the output image
# cv2.imshow("Min-Enclosing Circles", clone)
# cv2.waitKey(0)
perimax=0;
for (i, c) in enumerate(cnts):
	# compute the area and the perimeter of the contour
	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True)
	print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))
	if(perimax<perimeter):
		perimax=perimeter
		valuec=c
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		# cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
	# draw the contour on the image
	# cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
 
	# compute the center of the contour and draw the contour number
	# M = cv2.moments(c)
	# cX = int(M["m10"] / M["m00"])
	# cY = int(M["m01"] / M["m00"])
	# cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
	# 	1.25, (255, 255, 255), 4)
 
# show the output image
cv2.drawContours(gray, [valuec], -1, (0), 20)
cv2.imshow("Contours", gray)
cv2.waitKey(0)
clone = gray.copy()


(x, y, w, h) = cv2.boundingRect(valuec)
body = clonecircle[y:y+h,x:x+w]
cv2.imshow("Body", body)
cv2.waitKey(0)
# mask = np.zeros(clonecircle.shape[:2], dtype="uint8")
# cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
# clonecircle = body.copy()
cv2.rectangle(clonecircle, (x, y), (x + w, y + h), ( 255), 2)
cv2.line(clonecircle, (cX, cY), (cX + w, cY), 255)
w1=w
cv2.imshow("Bounding Boxes", clonecircle)
cv2.waitKey(0)

mask = np.zeros(clonecircle.shape[:2], dtype="uint8")
cv2.rectangle(mask, (x+5, y+5), (x + w-5, y + h-5), 255, -1)
cv2.imshow("Mask", mask)
cv2.waitKey(0)

clone = cv2.bitwise_and(clone,mask)
cv2.imshow("Tand",clone)
cv2.waitKey(0)

cloneaftercircle = image.copy()
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
	Xp = int(M["m10"] / M["m00"])
	Yp = int(M["m01"] / M["m00"])
	cv2.line(clonecircle, (cX, cY), (cX + 59, cY), 255)
	cv2.circle(clone,  (cX,cY), 10, ( 255), -1)
	cv2.imshow("Bounding Boxes10", clone)
	cv2.waitKey(0)
	cv2.circle(clone,  (cX+w+700,cY),15, ( 255), -1)
	cv2.imshow("Bounding Boxes10", clone)
	cv2.waitKey(0)
	cv2.circle(clone,  (Xp,Yp), 50, ( 255), -1)
	cv2.imshow("Bounding Boxes10", clone)
	cv2.waitKey(0)
	cXw=cX+w
	cv2.line(clonecircle, (cX, cY), (Xp, Yp), 255)
	p12 = math.sqrt(((cX-Xp)**2) + ((cY-Yp)**2)) #https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
	p13 = math.sqrt(((cX-cXw)**2) + ((cY-cY)**2))
	p23 = math.sqrt(((Xp-cXw)**2) + ((Yp-cY)**2))
	angle = math.degrees(math.acos((p12**2 + p13**2 - p23**2) / (2 * p12 * p13)))
# if fcount==1:
# 			fpos=char
# 			ftmp=result
# 		elif fcount>1:
	# print("angle {}".format(angle))
# double result = atan2(P3.y - P1.y, P3.x - P1.x) -
#                 atan2(P2.y - P1.y, P2.x - P1.x);

	result = math.degrees(math.atan2(cY - cY, cXw - cX) - math.atan2(Yp - cY, Xp - cY))
	
	if(result<0):
		result = 360+result
	# print("angle1 {} distance {}".format(result,p12))
	# cv2.imshow("Bounding Boxes", clonecircle)
	# cv2.waitKey(0)
	
	char = "?"

	if solidity > 0.9:
		char = "O"#+chr(a)

	elif solidity > 0.15:
		char = "X"#+chr(a)

	if char != "?":
		cv2.drawContours(image,[c],-1,(0,255,0),3)
		cv2.putText(image, char ,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,255,0),2)

	print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))

	if(((result>=0 and result<20) or (result>=340 and result<360)) and p12>35):
		fpos=char
	elif(result>=20 and result<75 and p12>35):
		cpos=char
	elif(result>=75 and result<110 and p12>35):
		bpos=char
	elif(result>=110 and result<160 and p12>35):
		apos=char
	elif(result>=160 and result<200 and p12>35):
		dpos=char
	elif(result>=200 and result<250 and p12>35):
		gpos=char
	elif(result>=250 and result<290 and p12>35):
		hpos=char
	elif(result>=290 and result<340 and p12>35):
		ipos=char
	elif(p12<35):
		epos=char
	print("{}|{}|{}".format(apos,bpos,cpos))
	print("--+--+--")
	print("{}|{}|{}".format(dpos,epos,fpos))
	print("--+--+--")
	print("{}|{}|{}".format(gpos,hpos,ipos))
	print("angle1 {} distance {}".format(result,p12))
	cv2.imshow("Bounding Boxes1", clonecircle)
	cv2.waitKey(0)
	
	# print
    # print apos,"|",bpos,"|",cpos
    # print("{}|{}|{}".format(apos,bpos,cpos))
cv2.imshow("Output3", image)
cv2.waitKey(0)
