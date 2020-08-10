import cv2

img = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg')
mask = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\BlueLongB_1_1_3.jpg',0)
res = cv2.bitwise_and(img,img,mask = mask)
cv2.imshow('resutt',res)
cv2.waitKey(0)