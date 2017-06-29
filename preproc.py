# Image Preprocessing

import numpy as np
import scipy
import os
from scipy.misc import imread
from scipy.misc import imresize
from scipy import ndimage
import cv2
from skimage import color
from skimage import io
from skimage import data, exposure, img_as_float




## SHARPENING ##

s1=['original1.png','original2.png','original3.png','original4.png','original5.png','original6.png']
s2=['pro1.png','pro2.png','pro3.png','pro4.png','pro5.png','pro6.png']
i=0




for fname in os.listdir('/usr2/prouserdata/surya/TrackBali/glyph_train1'):
        img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
        img_resize = cv2.resize(img,(28,28))
        scipy.misc.imsave(s1[i],img_resize)
	img1 = cv2.bilateralFilter(img,9,10,10)
        gamma_corrected = exposure.adjust_gamma(img, 1.5)
        #img_grayscale=color.rgb2gray(gamma_corrected)		
        img_resize = cv2.resize(gamma_corrected,(28,28))
        scipy.misc.imsave(s2[i],img_resize)
	i=i+1
	if i==5:
		break
