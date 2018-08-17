# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:13:00 2018

@author: Manuel
"""

import os
import numpy as np
from PIL import Image


#Iterate over all images and extract those with the correct resolution

i = 0

for root, dirs, files in os.walk("iaprtc12\\images"):
   for name in files:
      im = Image.open(os.path.join(root, name))
      if (im.size[0] == 360) and (im.size[1] == 480):
          #im.save("extractedImages\\"+str(i)+".jpg")
          print(i)
          i = i+1