import cv2 
import numpy as np
import matplotlib.pyplot as plt

window_name = 'image'
coor = []


def rgb2yiq():
    #https://github.com/hbtsai/rgb2yiq/blob/master/main.cpp
    pass

def kmean(image,k):
    #https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python#:~:text=Advertise-,How%20to%20Use%20K%2DMeans%20Clustering%20for%20Image%20Segmentation%20using,easier%20and%20more%20meaningful%20image.
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values) # convert to float
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)   # define stopping criteria
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)   # number of clusters (k)
    centers = np.uint8(centers) # convert back to 8 bit values
    labels = labels.flatten()       # flatten the labels array
    segmented_image = centers[labels.flatten()] # convert all pixels to the color of the centroids
    segmented_image = segmented_image.reshape(image.shape)  # reshape back to the original image dimension
    return segmented_image

def createTrackbar():
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html
    cv2.createTrackbar('high_H',window_name,0,255,nothing)
    cv2.createTrackbar('high_S',window_name,0,255,nothing)
    cv2.createTrackbar('high_V',window_name,0,255,nothing)
    cv2.createTrackbar('low_H',window_name,0,255,nothing)
    cv2.createTrackbar('low_S',window_name,0,255,nothing)
    cv2.createTrackbar('low_V',window_name,0,255,nothing)

    high_H = cv2.getTrackbarPos('high_H',window_name)
    high_S = cv2.getTrackbarPos('high_S',window_name)
    high_V = cv2.getTrackbarPos('high_V',window_name)
    low_H = cv2.getTrackbarPos('low_H',window_name)
    low_S = cv2.getTrackbarPos('low_S',window_name)
    low_V = cv2.getTrackbarPos('low_V',window_name)
    low_hsv = np.array([low_H,low_S,low_V])
    high_hsv = np.array([high_H,high_S,high_V])
    return low_hsv,high_hsv

def nothing(x):
    pass


def denoise(img,kernelSize):
    #https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/

    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=10) 
    img_dilation = cv2.dilate(img, kernel, iterations=10) 

def binaryMask():
    #https://answers.opencv.org/question/228538/how-to-create-a-binary-mask-for-medical-images/
    pass

def drawBoundingBox(thresh,original,image,ROI_number = 0):
    #https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    coor = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1
        x_min = x
        y_min = y
        x_max = x+w
        y_max = y+h
        #coor.append([x_min,y_min,x_max,y_max])
        print(ROI_number)
    return coor

def getCoordonateOfBB():
    pass

def writeToFile():
    pass