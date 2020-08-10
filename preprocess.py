import numpy as np
import cv2
import matplotlib.pyplot as plt
import myFunction


window_name = "image"
coor = []

def process(img,imgpath,i):
    window_name = 'image_'+str(i)+'   '+imgpath
    original = img.copy()
    kernel = np.ones((3,3), np.uint8)
    # dilated_img = cv2.dilate(img, kernel, iterations=2) 
    # eroded_img = cv2.erode(original, kernel, iterations=2)
    segmented_image = myFunction.kmean(img,2)
    
    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    #low_hsv,high_hsv = myFunction.createTrackbar()
    lower_hsv, upper_hsv = np.array([0,0,0]),np.array([180,180,150])
    blur = cv2.GaussianBlur(hsv,(5,5),0)
    thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)
    thresh = cv2.bitwise_not(thresh)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,img)
    #cv2.imwrite("image.jpg",segmented_image)
    #cv2.imshow('dilated',dilated_img)
    #cv2.imshow('eroded',eroded_img)
    #cv2.imshow('thresh',thresh)
    return coor
def processRedBrown(img,imgpath,i):
    window_name = 'image_'+str(i)+'   '+imgpath
    original = img.copy()
    kernel = np.ones((3,3), np.uint8)
    # dilated_img = cv2.dilate(img, kernel, iterations=2) 
    # eroded_img = cv2.erode(original, kernel, iterations=2)
    segmented_image = myFunction.kmean(img,2)
    
    # hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    # #low_hsv,high_hsv = myFunction.createTrackbar()
    # lower_hsv, upper_hsv = np.array([0,0,0]),np.array([180,180,150])
    # blur = cv2.GaussianBlur(hsv,(5,5),0)
    # thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # thresh = cv2.bitwise_not(thresh)
    thresh = myFunction.detectRed(segmented_image)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,segmented_image)
    #cv2.imwrite("image.jpg",segmented_image)
    #cv2.imshow('dilated',dilated_img)
    #cv2.imshow('eroded',eroded_img)
    #cv2.imshow('thresh',thresh)
    return coor
    

def processRed(img):
    original = img.copy()
    thresh = myFunction.detectRed(img)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,img)
    #cv2.imshow('thresh',thresh)
    return coor


def processOrange(img):
    original = img.copy()
    blur = myFunction.denoise(original,10)
    thresh = myFunction.detectOrange(img)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,img)
    #cv2.imshow('thresh',thresh)
    return coor

def processNearRed(img):
    original = img.copy()
    thresh = myFunction.detectRedPink(img)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,img)
    #cv2.imshow('thresh',thresh)
    return coor
    
    
# img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test', 1)
# coor = process(img)
# cv2.imshow('result',img)
# cv2.waitKey(0)

# img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\GreenSmallBoardT_1_1_6.jpg', 1)
# original = img.copy()
# myFunction.createTrackbar('trackbar')
# myFunction.denoise(img,5)
# while(1):
#     img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\GreenSmallBoardT_1_1_6.jpg', 1)
#     low_hsv,high_hsv = myFunction.getTrackbarPos('trackbar')
#     thresh = myFunction.detectRedPink(img,high_hsv,low_hsv)
#     coor = myFunction.drawBoundingBox(thresh,original,img)
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()