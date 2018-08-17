# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:52:25 2018

@author: Manuel
"""
from enum import IntEnum
import numpy as np
from skimage import io, color

EPSILON = 0.000001

# enumeration for the orientations used to determine dissimilarity
# between pieces
class Orientations(IntEnum):
    left = 1
    right = 2
    up = 3
    down = 4
    
class Piece:
    
    NeighborRight = []
    NeighborLeft= []
    NeighborUp = []
    NeighborDown = []
    
    
    def __init__(self, data, NeighborRight, NeighborLeft, NeighborUp, NeighborDown):
        #Initialize a piece with the image data and true neighboring pieces
        self.data = data
        self.trueNeighborRight = NeighborRight
        self.trueNeighborLeft = NeighborLeft
        self.trueNeighborUp = NeighborUp
        self.trueNeighborDown = NeighborDown



# returns the slices of the pieces that have to be used to calculate
# the dissimilarity
def slices(pi, pj,  orientation):
    K = pi.shape[0] - 1
    if orientation == Orientations.right:
        return pi[:, K, :], pi[:, (K-1), :], pj[:, 0, :]
    if orientation == Orientations.left:
        return pi[:, 0, :], pi[:, 1, :], pj[:, K, :]
    if orientation == Orientations.up:
        return pi[0, :, :], pi[1, :, :], pj[K, :, :]
    if orientation == Orientations.down:
        return pi[K, :, :], pi[(K-1), :, :], pj[0, :, :]
    

def dissmiliarity(pi, pj, orientation):
    slice1, slice2, slice3 = slices(pi, pj, orientation)
    dissim = np.sqrt(np.sum((slice1 - slice3)**2))
    if dissim == 0.0:
        return EPSILON
    return dissim

def compatibility(pi, pj, orientation):
    dissimilarityPiPj = dissmiliarity(pi, pj, orientation)
    return -dissimilarityPiPj


# crops piece out of given image
def crop(infile, height, width):
    im = io.imread(infile)
    imgwidth, imgheight = im.shape[1], im.shape[0]
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            yield im[(i*height):((i+1)*height), (j*width):((j+1)*width), :]


# cuts given image into pieces and returns list containing the pieces
# pieces are in LAB color space
def cutIntoPieces(infile, height, width):
    pieces = []
    for k, piece in enumerate(crop(infile, height, width)):
        img = color.rgb2lab(piece)
        pieces.append(img)
    return pieces



pieces = cutIntoPieces("iaprtc12/images/00/25.jpg", 28, 28)
pieceList = []

# Create Piece objects from the original data
for p in pieces:
    pi = Piece(p,[],[],[],[])
    pieceList.append(pi)
    
    
#Test - there are three nested for-loops, I know it's awful
compMat = np.ones((204,203,4))

for i,pi in enumerate(pieceList):
    for k,pj in enumerate([x for j,x in enumerate(pieceList) if j!=i]):
        for s,orientation in enumerate(Orientations):
            compMat[i,k,s] = compatibility(pi.data,pj.data,orientation)
    
    
    
    
    