#%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('test.jpg')
copy_img = cv.imread('test.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
copy_img = cv.cvtColor(copy_img,cv.COLOR_BGR2RGB)
lower = np.array([20,20,20])
upper = np.array([250,250,250])
mask = cv.inRange(img,lower,upper)
cont, _ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
count_image = cv.drawContours(img,cont,-1,255,3)
c = max(cont,key = cv.contourArea)
x,y,w,h = cv.boundingRect(c)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
cropped_image = img[y:y+h,x:x+w]
#plt.imshow(img)
#plt.imshow(mask,'gray')
#plt.imshow(count_image)
#plt.imshow(cropped_image)
plt.figure(figsize=(20,4))
plt.subplot(1,3,1),plt.imshow(copy_img)
plt.subplot(1,3,2),plt.imshow(img)
plt.subplot(1,3,3),plt.imshow(cropped_image)
plt.show()
