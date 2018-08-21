# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:13:00 2018

@author: Manuel
"""

import os
import numpy as np
from PIL import Image

np.random.seed(200)


#Iterate over all images and extract those with the correct resolution
#Since there are 6000 images with this resolution the feature vectors would contain
#several millions of instances. We choose a subset randomly to get ~1000 images

i = 0

for root, dirs, files in os.walk("iaprtc12\\images"):
   for name in files:
      im = Image.open(os.path.join(root, name))
      randomThreshold = 0.2
      if (np.random.random() < randomThreshold) and (im.size[0] == 480) and (im.size[1] == 360):
          im.save("extractedImages\\"+str(i)+".jpg")
          #print(i)
          i = i+1