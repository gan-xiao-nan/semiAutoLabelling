# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import myFunction
import os
import preprocess 
import cv2

coor = []
folder_path = r'C:\Users\xiao-nan.gan\Desktop\autoLabel\images'

for path in os.listdir(folder_path):#loop to read one image at a time 
    if path.endswith('.jpg'):
        imgpath = os.path.join(folder_path, path)
        # renamee is the file getting renamed, pre is the part of file name before extension and ext is current extension
        pre, ext = os.path.splitext(imgpath)
        txtpath = pre + '.txt'
        print(imgpath)
        img = cv2.imread(imgpath, 1)
        coor = preprocess.process(img)
        key = cv2.waitKey(5000)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

        elif key == ord('s'): # wait for 's' key to save and exit
            #cv2.imwrite('messigray.png',img)
            print(type(coor),len(coor))
            with open(txtpath, "w") as outfile:
                outfile.write("start here \n")
                for x in range(0,len(coor)):
                    for y in range(0,len(coor[x])):
                        outfile.write(str(output[x]))
                        outfile.write(',')
                    outfile.write('PAD','\n')
                outfile.close()
            coor = []
            cv2.destroyAllWindows()