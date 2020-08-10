import cv2
import myFunction
import numpy as np
import myFunction

myFunction.createTrackbar('trackbar')
max_H = 0
image = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg', cv2.IMREAD_UNCHANGED)
#image2 = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg', 0)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
segmented_image,centers = myFunction.kmean(img_hsv,2)
for x in range(len(centers)):
    print('maxH = ' , max_H)
    if centers[x][2]> max_H:
        lower_hsv = centers[x]
        max_H = centers[x][2]
    else:
        pass
print('hsv = ',int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2]))
kmean_mask = cv2.inRange(segmented_image,(int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2])),(255,255,255))

masked_segmented = cv2.bitwise_and(img_hsv,img_hsv,mask = kmean_mask)
cv2.imshow('masked_segmented',masked_segmented)
cv2.imshow('original',img_hsv)

low_hsv,high_hsv = myFunction.getTrackbarPos('trackbar')
red_mask = cv2.inRange(masked_segmented,(int(low_hsv[0]),int(low_hsv[1]),int(low_hsv[2])),(int(high_hsv[0]),int(high_hsv[1]),int(high_hsv[2])))
withHole = cv2.bitwise_and(image,image,mask = red_mask)
noHole = myFunction.denoise(withHole,7)
cv2.imshow('red_mask',red_mask)
cv2.imshow('mask',kmean_mask)

cv2.imshow('withHole',withHole)

#convert img to grey
img_grey = cv2.cvtColor(noHole,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
# #find contours
# cv2.imshow('thresh_image',thresh_img)
# cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in cnts: 
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) 
#     if len(approx)>=5: 
#         print ("pentagon")
#         cv2.drawContours(noHole,[cnt],0,(0,0,255),-1) 
#     elif len(approx)==3: 
#         print ("triangle")
#         cv2.drawContours(noHole,[cnt],0,(0,0,0),-1) 
#     elif len(approx)==4: 
#         print ("square")
#         cv2.drawContours(noHole,[cnt],0,(255,0,0),-1) 
#     elif len(approx) == 9: 
#         print ("half-circle")
#         cv2.drawContours(noHole,[cnt],0,(125,125,125),-1) 
#     elif len(approx) > 15: 
#         print ("circle")
#         cv2.drawContours(noHole,[cnt],0,(0,255,0),-1) 
myFunction.drawBoundingBox(thresh_img,img_hsv,image)
cv2.imshow('noHole',noHole)
cv2.imshow('image',image)


key = cv2.waitKey(0)#pauses for 3 seconds before fetching next image
if key == 27:#if ESC is pressed, exit loop
    cv2.destroyAllWindows()


