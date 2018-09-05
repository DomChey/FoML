# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:52:25 2018

@author: Manuel
"""
import numpy as np
from skimage import io, color
import os
from tqdm import tqdm
from imgCrop import cutIntoPieces, createPieces
from accessory import Orientations
from compatibility import compatibility

np.random.seed(100)

def createTrainingData(file):
    # Create training data from one single image file which is split into
    # 12 x 17 tiles.
    # Returns several positive and negative instances from this image
    pieces = cutIntoPieces(file, 28, 28)
    pieceList = createPieces(pieces)
    
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
                
    negLabeled, posLabeled = np.array(negLabeled), np.array(posLabeled)
    
    #reduce the negativeFeatures for a balanced set
    randomMask = np.random.choice(negLabeled.shape[0],posLabeled.shape[0], replace = False)
    negLabeled = negLabeled[randomMask,:,:,:]
    
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


if __name__ == "__main__":
    # Get the features from all the extracted images
    
    positiveFeatures, negativeFeatures = scanImagesForTraining("extractedImages/")
    
    positiveFeatures, negativeFeatures = np.array(positiveFeatures), np.array(negativeFeatures)
    
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
