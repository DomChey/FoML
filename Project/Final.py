from imgCrop import cutIntoPieces
from Placer import placer, getImage, getShuffledImage
import numpy as np
import matplotlib.pyplot as plt
from compatibility import *
from FindStartingPiece import *

def clearAllMemoizedFunctions():
    dissmiliarity.clearMemo()
    secondBestDissmilarity.clearMemo()
    areBestBuddies.clearMemo()
    bestBuddy.clearMemo()
    mutualCompatibility.clearMemo()
    hasFourBB.clearMemo()

# Save all results from the whole dataset
def solvePuzzle(i):

    pieces = cutIntoPieces("imData/{}.png".format(i), 24, 24)
    pieces = np.array(pieces)
    np.random.shuffle(pieces)
    pieces = list(pieces)
    shuffledImage = getShuffledImage(pieces, 672, 504)
    plt.imsave("results/{}_shuffled.png".format(i), shuffledImage)
    
    sort = placer(pieces)
    finalImage = getImage(sort)
    plt.imsave("results/{}_solved.png".format(i), finalImage)
    clearAllMemoizedFunctions()


for i in range(1,21):
    solvePuzzle(i)

#solvePuzzle(4)
