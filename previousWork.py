import cv2
import myFunction
import numpy as np
import myFunction

def blueLong(image):
    myFunction.createTrackbar('trackbar')
    #image = cv2.imread(imgpath)
    #image2 = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg', 0)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_segmented = myFunction.kmean_mask(img_hsv,2)

    low_hsv,high_hsv = myFunction.getTrackbarPos('trackbar')
    red_mask = cv2.inRange(masked_segmented,(int(low_hsv[0]),int(low_hsv[1]),int(low_hsv[2])),(int(high_hsv[0]),int(high_hsv[1]),int(high_hsv[2])))
    withHole = cv2.bitwise_and(image,image,mask = red_mask)
    noHole = myFunction.denoise(withHole,7)

    #convert img to grey
    img_grey = cv2.cvtColor(noHole,cv2.COLOR_BGR2GRAY)
    thresh = 100
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    coor = myFunction.drawBoundingBox(thresh_img,img_hsv,image)
    #cv2.imshow('noHole',noHole)
    cv2.imshow('image',image)
    return coor


    # key = cv2.waitKey(0)#pauses for 3 seconds before fetching next image
    # if key == 27:#if ESC is pressed, exit loop
    #     cv2.destroyAllWindows()


