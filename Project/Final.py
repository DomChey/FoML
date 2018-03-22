from imgCrop import cutIntoPieces
from Placer import placer, getImage, getShuffledImage
import time
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
def solvePuzzle(i, log):
    res = 80
    sourcePieces = cutIntoPieces("imData/{}.png".format(i), res, res)
    pieces = np.array(sourcePieces)
    np.random.shuffle(pieces)
    pieces = list(pieces)
    shuffledImage = getShuffledImage(pieces, 672, 504)
    plt.imsave("results/{}_shuffled.png".format(i), shuffledImage)
    
    start_time = time.time()
    sort = placer(pieces)
    elapsed_time = time.time() - start_time
    
    finalImage = getImage(sort)
    plt.imsave("results/{}_solved.png".format(i), finalImage)
    clearAllMemoizedFunctions()
    
    absoluteScore = absoluteEval(sourcePieces, sort)
    
    log.write("Image {} with {} pieces\nElapsed time: {:.4f}s\nAbsolute Score: {:.3f}%\n\n"
              .format(i, len(pieces), elapsed_time, absoluteScore*100))



log = open("results/results_log.txt", "w")

for i in range(1,21):   
    solvePuzzle(i, log)

log.close()

