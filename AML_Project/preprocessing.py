# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:52:25 2018

@author: Manuel
"""
from enum import IntEnum
import numpy as np
from skimage import io, color
import os
from tqdm import tqdm

np.random.seed(100)

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
def slices(pi, pj, orientation):
    K = pi.shape[0] - 1
    if orientation == Orientations.right:
        return pi[:, K, :], pi[:, (K-1), :], pj[:, 0, :], pj[:,1,:]
    if orientation == Orientations.left:
        return pi[:, 0, :], pi[:, 1, :], pj[:, K, :], pj[:,(K-1),:]
    if orientation == Orientations.up:
        return pi[0, :, :], pi[1, :, :], pj[K, :, :], pj[(K-1),:,:]
    if orientation == Orientations.down:
        return pi[K, :, :], pi[(K-1), :, :], pj[0, :, :], pj[1,:,:]
    

def dissmiliarity(pi, pj, orientation):
    slice1, slice2, slice3, slice4 = slices(pi, pj, orientation)
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
        img = color.rgb2yuv(piece)
        pieces.append(img)
    return pieces


def createTrainingData(file):
    # Create training data from one single image file which is split into
    # 12 x 17 tiles.
    # Returns several positive and negative instances from this image
    pieces = cutIntoPieces(file, 28, 28)
    pieceList = []
    
    # Create Piece objects from the original data
    # Convention: x axis starts from the upper left corner to the right
    # y axis starts from the upper left corner downwards
    for i,p in enumerate(pieces):
        xpos = i%12
        ypos = int(np.floor(i/12))
        
        if xpos == 0 and ypos == 0:
            neighborLeft = []
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = pieces[xpos+1]
        elif xpos == 11 and ypos == 0:
            neighborRight = []
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12+xpos]
            neighborLeft = pieces[xpos-1]
        elif xpos == 0 and ypos == 16:
            neighborRight = pieces[ypos*12]
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = []
            neighborLeft = []
        elif xpos == 11 and ypos == 16:
            neighborRight = []
            neighborUp = pieces[(ypos-1)*12 + xpos]
            neighborDown = []
            neighborLeft = pieces[(ypos)*12 + xpos-1]
        elif xpos == 0:
            neighborLeft = []
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = pieces[ypos*12 + xpos+1]
        elif xpos == 11:
            neighborLeft = pieces[ypos*12 + xpos-1]
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = []
        elif ypos == 0:
            neighborRight = pieces[xpos+1]
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12+xpos]
            neighborLeft = pieces[xpos-1]
        elif ypos == 16:
            neighborRight = pieces[ypos*12 + xpos+1]
            neighborUp = pieces[(ypos-1)*12+xpos]
            neighborDown = []
            neighborLeft = pieces[ypos*12 + xpos-1]
        else:
            neighborRight = pieces[ypos*12 + xpos+1]
            neighborUp = pieces[(ypos-1)*12 + xpos]
            neighborDown = pieces[(ypos+1)*12 + xpos]
            neighborLeft = pieces[ypos*12 + xpos-1]
        
        pi = Piece(p, neighborRight, neighborLeft, neighborUp, neighborDown)
        pieceList.append(pi)
        
        
    compMat = np.ones((204,204,4))
    
    for i,pi in enumerate(pieceList):
        for k,pj in enumerate(pieceList):   #[x for j,x in enumerate(pieceList) if j!=i]):
            for s,orientation in enumerate(Orientations):
                compMat[i,k,s] = compatibility(pi.data,pj.data,orientation)
        
    
    posLabeled = []
    negLabeled = []    
    
    for i,pi in enumerate(pieceList):
        for s,orientation in enumerate(Orientations):
            sorting = np.argsort(compMat[i,:,s])
            if sorting[-1] == i:
                idx = sorting[-2]
                idx2 = sorting[-3]
            else:
                idx = sorting[-1]
                idx2 = sorting[-2]
    
            if orientation == Orientations.up:
                data = pi.trueNeighborUp
            elif orientation == Orientations.down:
                data = pi.trueNeighborDown
            elif orientation == Orientations.left:
                data = pi.trueNeighborLeft
            elif orientation == Orientations.right:
                data = pi.trueNeighborRight
            
            if data != [] and (np.equal(pieceList[idx].data, data)).all():
                #print("The most compatible piece is the true neighbor")
                s1, s2, s3, s4 = slices(pi.data, pieceList[idx].data, orientation)
                posLabeled.append( np.stack((s2,s1,s3,s4), axis=1) )
                
                s1, s2, s3, s4 = slices(pi.data, pieceList[idx2].data, orientation)
                negLabeled.append( np.stack((s2,s1,s3,s4), axis=1) )
            else:
                #print("The most compatible piece is not the true neighbor")
                s1, s2, s3, s4 = slices(pi.data, pieceList[idx2].data, orientation)
                negLabeled.append( np.stack((s2,s1,s3,s4), axis=1) )
                
    return posLabeled, negLabeled


def scanImagesForTraining(rootdir):
    # Scans all images in rootdir and extracts the features for the neural net
    # (One instance is a 28x4x3 matrix)
    positiveFeatures = []
    negativeFeatures = []
    
    for root, dirs, files in os.walk(rootdir):
        for image in tqdm(files):
            posLabeled, negLabeled = createTrainingData(root + image)
            positiveFeatures.extend(posLabeled)
            negativeFeatures.extend(negLabeled)
    return positiveFeatures, negativeFeatures
            


# Get the features from all the extracted images

positiveFeatures, negativeFeatures = scanImagesForTraining("extractedImages/")

positiveFeatures, negativeFeatures = np.array(positiveFeatures), np.array(negativeFeatures)

#reduce the negativeFeatures for a balanced set
randomMask = np.random.choice(negativeFeatures.shape[0],positiveFeatures.shape[0], replace = False)
negativeFeatures = negativeFeatures[randomMask,:,:,:]

import gzip

f = gzip.GzipFile("positiveFeatures.npy.gz", "w")
np.save(file = f, arr=positiveFeatures)
f.close()

f = gzip.GzipFile("negativeFeatures.npy.gz", "w")
np.save(file = f, arr=negativeFeatures)
f.close()

# To load the arrays: 
# f = gzip.GzipFile("positiveFeatures.npy.gz", "r")
# array = np.load(f)