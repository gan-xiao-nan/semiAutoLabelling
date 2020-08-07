

#!/usr/bin/python3
# 2018.07.08 10:39:15 CST
# 2018.07.08 11:09:44 CST
import cv2
import numpy as np
## Read and merge
img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\SMCAMTop_3_1_7.jpg',1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Gen lower mask (0-5) and upper mask (175-180) of RED
mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2 )
croped = cv2.bitwise_and(img, img, mask=mask)

## Display
cv2.imshow("mask", mask)
cv2.imshow("croped", croped)
cv2.waitKey()