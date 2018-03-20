from imgCrop import cutIntoPieces
from Placer import placer, getImage, getShuffledImage
import numpy as np
import matplotlib.pyplot as plt



# Save all results from the whole dataset
for i in range(1,21):

    pieces = cutIntoPieces("imData/{}.png".format(i), 50, 50)
    pieces = np.array(pieces)
    np.random.shuffle(pieces)
    pieces = list(pieces)
    shuffledImage = getShuffledImage(pieces, 672, 504)
    plt.imsave("results/{}_shuffled.png".format(i), shuffledImage)
    
    sort = placer(pieces)
    finalImage = getImage(sort)
    plt.imsave("results/{}_solved.png".format(i), finalImage)


