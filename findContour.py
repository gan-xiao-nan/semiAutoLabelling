import cv2
import numpy as np

img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg', cv2.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
cv2.imshow('thresh_image',thresh_img)
cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # #create an empty image for contours
# # img_contours = np.zeros(img.shape)
# # # draw the contours on the empty image
# # cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# rect = cv2.minAreaRect(cnts[0])
# box = np.int0(cv2.boxPoints(rect))
# cv2.drawContours(img, [box], 0, (36,255,12), 3) # OR

# #save image
# cv2.imshow('D:/contours.png',img) 
# cv2.waitKey(0)

for cnt in cnts: 
    approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True) 
    if len(approx)==5: 
        print ("pentagon")
        cv2.drawContours(img,[cnt],0,255,-1) 
    elif len(approx)==3: 
        print ("triangle")
        cv2.drawContours(img,[cnt],0,(0,255,0),-1) 
    elif len(approx)==4: 
        print ("square")
        cv2.drawContours(img,[cnt],0,(255,255,255),-1) 
    elif len(approx) == 9: 
        print ("half-circle")
        cv2.drawContours(img,[cnt],0,(255,255,0),-1) 
    elif len(approx) > 15: 
        print ("circle")
        cv2.drawContours(img,[cnt],0,(0,0,0),-1) 
 
cv2.imshow('img',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()