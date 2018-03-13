from imgCrop import *
import numpy as np

pieces = cutIntoPieces("c:\\Users\\Manuel\\OneDrive\\Machine Learning\\FoML\\Project\\imData\\1.png", 56, 56)

pieces = np.array(pieces)
dim = pieces.shape

print(dim)