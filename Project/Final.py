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
def solvePuzzle(i, log, numPieces):
    if numPieces == 432:
        imWidth = 672
        imHeight = 504
        maxCol = 23
        maxRow = 17
        imFormat = "png"
    elif numPieces == 540:
        imWidth = 756
        imHeight = 560
        maxCol = 26
        maxRow = 19
        imFormat = "jpg"
    elif numPieces == 805:
        imWidth = 980
        imHeight = 644
        maxCol = 34
        maxRow = 22
        imFormat = "jpg"
    elif numPieces == 2360:
        imWidth = 1652
        imHeight = 1120
        maxCol = 58
        maxRow = 39
        imFormat = "jpg"

    np.random.seed(2017)
    random.seed(2017)
    res = 28
    sourcePieces = cutIntoPieces("imData/{}/{}.{}".format(numPieces, i, imFormat), res, res)
    pieces = np.array(sourcePieces)
    np.random.shuffle(pieces)
    pieces = list(pieces)
    shuffledImage = getShuffledImage(pieces, imWidth, imHeight)
    plt.imsave("results/{}/{}_shuffled.png".format(numPieces, i), shuffledImage)
    
    start_time = time.time()
    sort = placer(pieces, maxCol, maxRow)
    elapsed_time = time.time() - start_time
    
    finalImage = getImage(sort)
    plt.imsave("results/{}/{}_solved.png".format(numPieces, i), finalImage)
    clearAllMemoizedFunctions()
    
    absoluteScore = absoluteEval(sourcePieces, sort)
    print("Absolute score for image {} with {} pieces is: {:.3f}".format(i, numPieces, absoluteScore*100))
    
    log.write("Image {} with {} pieces\nElapsed time: {:.4f}s\nAbsolute Score: {:.3f}%\n\n"
              .format(i, len(pieces), elapsed_time, absoluteScore*100))




log = open("results/results_log.txt", "w")
totalScore = 0
for i in range(3,4):   
    solvePuzzle(i, log, 2360)
#solvePuzzle(4, log, 432)

log.close()

