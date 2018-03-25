from imgCrop import cutIntoPieces
from Placer import placer, getImage, getShuffledImage
import time, random
import numpy as np
import matplotlib.pyplot as plt
from compatibility import *
from FindStartingPiece import *
from Evaluation import absoluteEval

def clearAllMemoizedFunctions():
    dissmiliarity.clearMemo()
    secondBestDissmilarity.clearMemo()
    areBestBuddies.clearMemo()
    bestBuddy.clearMemo()
    mutualCompatibility.clearMemo()
    hasFourBB.clearMemo()

# Save all results from the whole dataset
def solvePuzzle(i, log, maxCol, maxRow):
    res = 28
    sourcePieces = cutIntoPieces("imData/{}.png".format(i), res, res)
    pieces = np.array(sourcePieces)
    np.random.shuffle(pieces)
    pieces = list(pieces)
    shuffledImage = getShuffledImage(pieces, 672, 504)
    plt.imsave("results/{}_shuffled.png".format(i), shuffledImage)
    
    start_time = time.time()
    sort = placer(pieces, maxCol, maxRow)
    elapsed_time = time.time() - start_time
    
    finalImage = getImage(sort)
    plt.imsave("results/{}_solved.png".format(i), finalImage)
    clearAllMemoizedFunctions()
    
    absoluteScore = absoluteEval(sourcePieces, sort)
    
    log.write("Image {} with {} pieces\nElapsed time: {:.4f}s\nAbsolute Score: {:.3f}%\n\n"
              .format(i, len(pieces), elapsed_time, absoluteScore*100))
    return absoluteScore


log = open("results/results_log.txt", "w")
np.random.seed(2017)
random.seed(2017)
totalScore = 0
#for i in range(1,21):   
#    totalScore += solvePuzzle(i, log, 23, 17)
solvePuzzle(15, log, 23, 17)
totalScore /= 20

log.write("Mean Absolute Score: {:.3f}%\n\n".format(totalScore*100))

log.close()

