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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)   # define stopping criteria
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)   # number of clusters (k)
    centers = np.uint8(centers) # convert back to 8 bit values
    print('centers',centers)
    labels = labels.flatten()       # flatten the labels array
    segmented_image = centers[labels.flatten()] # convert all pixels to the color of the centroids
    segmented_image = segmented_image.reshape(image.shape)  # reshape back to the original image dimension
    return segmented_image,centers

def createTrackbar(trackbar_name):
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html
    cv2.namedWindow(trackbar_name)
    cv2.createTrackbar('low_H',trackbar_name,0,180,nothing)
    cv2.createTrackbar('low_S',trackbar_name,0,255,nothing)
    cv2.createTrackbar('low_V',trackbar_name,0,255,nothing)
    cv2.createTrackbar('high_H',trackbar_name,0,180,nothing)
    cv2.createTrackbar('high_S',trackbar_name,0,255,nothing)
    cv2.createTrackbar('high_V',trackbar_name,0,255,nothing)

    cv2.setTrackbarPos('low_H', trackbar_name, 0)   #yellow-orange (0,14,185),(35,130,255)
    cv2.setTrackbarPos('low_S', trackbar_name, 14)  #pinkish, yellow also ok (0,14,199),(180,127,255)
    cv2.setTrackbarPos('low_V', trackbar_name, 193) #pink can, yellow cannot(0,62,188),(180,178,255)
    cv2.setTrackbarPos('high_H', trackbar_name, 180)
    cv2.setTrackbarPos('high_S', trackbar_name, 255)
    cv2.setTrackbarPos('high_V', trackbar_name, 255)


def getTrackbarPos(trackbar_name):
    high_H = cv2.getTrackbarPos('high_H',trackbar_name)
    high_S = cv2.getTrackbarPos('high_S',trackbar_name)
    high_V = cv2.getTrackbarPos('high_V',trackbar_name)
    low_H = cv2.getTrackbarPos('low_H',trackbar_name)
    low_S = cv2.getTrackbarPos('low_S',trackbar_name)
    low_V = cv2.getTrackbarPos('low_V',trackbar_name)
    low_hsv = np.array([low_H,low_S,low_V])
    high_hsv = np.array([high_H,high_S,high_V])
    return low_hsv,high_hsv

def nothing(x):
    pass

def detectRed(img):
    #https://stackoverflow.com/questions/51229126/how-to-find-the-red-color-regions-using-opencv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    denoise(img_hsv,3)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2 )    
    return mask

def detectOrange(img):
    #https://stackoverflow.com/questions/51229126/how-to-find-the-red-color-regions-using-opencv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    denoise(img_hsv,5)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask = cv2.inRange(img_hsv, (5,20,70), (35,255,255))
    #mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    ## Merge the mask and crop the red regions
    #mask = cv2.bitwise_or(mask1, mask2 )    
    return mask


def denoise(img,kernelSize):
    #https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/

    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    img = cv2.erode(img, kernel, iterations=1) 
    img = cv2.dilate(img, kernel, iterations=1) 
    img = cv2.erode(img, kernel, iterations=1) 
    img = cv2.dilate(img, kernel, iterations=1) 
    img = cv2.GaussianBlur(img,(5,5),0)
    return img


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
        if ((w*h)>200 and (w*h)<40000):
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            ROI_number += 1
            x_min = x
            y_min = y
            x_max = x+w
            y_max = y+h
            coor.append([x_min,y_min,x_max,y_max])
        else:
            pass
    return coor

def kmean_mask(img_hsv,k,max_H = 0):
    segmented_image,centers = kmean(img_hsv,2)
    for x in range(len(centers)):
        #print('maxH = ' , max_H)
        if centers[x][2]> max_H:
            lower_hsv = centers[x]
            max_H = centers[x][2]
        else:
            pass
    print('hsv = ',int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2]))
    kmean_mask = cv2.inRange(segmented_image,(int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2])),(255,255,255))
    masked_segmented = cv2.bitwise_and(img_hsv,img_hsv,mask = kmean_mask)
    # cv2.imshow('masked_segmented',masked_segmented)
    # cv2.imshow('original',img_hsv)
    return masked_segmented
