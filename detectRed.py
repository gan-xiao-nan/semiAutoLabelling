import numpy as np
import cv2
import matplotlib.pyplot as plt
import myFunction

image = cv2.imread(r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images\test\SMCAMTop_3_1_7.jpg')

# define the list of boundaries
boundaries = [
	([17, 15, 100], [50, 56, 200])
]
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", output)
	cv2.waitKey(0)