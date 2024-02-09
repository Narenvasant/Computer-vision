#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


#Defining the Filter 
def idealFilter(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
#reading the input image
img = cv2.imread("asguard2.png", 0)
plt.subplot(161), plt.imshow(img, "gray"), plt.title("Oringinal Image")
#applying fourier transform
original = np.fft.fft2(img)
plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
#applying fourier shift
center = np.fft.fftshift(original)
plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")
#center the frequency components
CenteredFilter = center * idealFilter(50,img.shape)
plt.subplot(164), plt.imshow(np.log(1+np.abs(CenteredFilter)), "gray"), plt.title("Centered Filter Spectrum")
#applying inverse fourier shift
Decentralize = np.fft.ifftshift(CenteredFilter)
plt.subplot(165), plt.imshow(np.log(1+np.abs(Decentralize)), "gray"), plt.title("Decentralize")
#applying inverse fourier transform to get the Reconstructed Image
inverse_shift = np.fft.ifft2(Decentralize)
plt.subplot(166), plt.imshow(np.abs(inverse_shift), "gray"), plt.title("Processed Image")
plt.show()


# In[ ]:




