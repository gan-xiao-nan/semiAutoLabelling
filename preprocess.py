import numpy as np
import cv2
import matplotlib.pyplot as plt
import myFunction


window_name = "image"
ROI_number = 0
coor = []

def process(img,imgpath,i):
    window_name = 'image_'+str(i)+'   '+imgpath
    original = img.copy()
    segmented_image = myFunction.kmean(img,2)
    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    #low_hsv,high_hsv = myFunction.createTrackbar()
    lower_hsv, upper_hsv = np.array([0,0,0]),np.array([180,180,180])
    blur = cv2.GaussianBlur(hsv,(5,5),0)
    thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)
    thresh = cv2.bitwise_not(thresh)
    coor = myFunction.drawBoundingBox(thresh,original,img)
    cv2.imshow(window_name,img)
    #cv2.imshow('thresh',thresh)
    return coor

    
# img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\49043Bottom_1_3_1.jpg', 1)
# coor = process(img,'path',2)
# cv2.imshow('result',img)
# cv2.waitKey(0)

