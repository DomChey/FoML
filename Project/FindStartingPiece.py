#Assumptions:
#1) function compatibility(p1,p2,r) calculates and returns the compatibility score of p1 and p2 in direction r
#2) function bestBuddy(p,r) calculates and returns the best buddy (if existent) in direction r
#3) numpy array 'pieces' contains all pieces in a specific order


import numpy as np

def mutualCompatibility(p1,p2,r):
    # Calculate the mutual compatibility of pieces p1 and p2 in direction r

    r1 = r
    if r1=='up':
        r2 = 'down'
    elif r1 == 'down':
        r2 = 'up'
    elif r1 == 'right':
        r2 = 'left'
    elif r1 == 'left':
        r2 = 'right'
    return ((compatibility(p1,p2,r1)+compatibility(p2,p1,r2))/2)


def hasFourBB(x):
    return(bestBuddy(x,'up') and
        bestBuddy(x,'down') and
        bestBuddy(x,'right') and
        bestBuddy(x,'left'))


def findFirstPiece(pieces):
    # Find a piece that has best buddies in all four spatial dimensions
    # and maximizes the mutual compatibility

    distinctivePieces = filter(lambda x: hasFourBB(x), pieces)

    piecesInDistinctiveRegion = np.array([])
    piecesInDistinctiveRegionBB = np.array([])

    for x in distinctivePieces:
        left = bestBuddy(x, 'left')
        right = bestBuddy(x, 'right')
        up = bestBuddy(x, 'up')
        down = bestBuddy(x, 'down')
        
        if (hasFourBB(left) and hasFourBB(right) and hasFourBB(up) and hasFourBB(down)):
            piecesInDistinctiveRegion.append(x)
            piecesInDistinctiveRegionBB.append([left,right,up,down])

    mutualComp = [ mutualCompatibility(x,piecesInDistinctiveRegionBB[i,0], 'left')+
    mutualCompatibility(x,piecesInDistinctiveRegionBB[i,1], 'right')+
    mutualCompatibility(x,piecesInDistinctiveRegionBB[i,2], 'up')+
    mutualCompatibility(x,piecesInDistinctiveRegionBB[i,3], 'down')
    for (i,x) in enumerate(piecesInDistinctiveRegion) ]

    return piecesInDistinctiveRegion[np.argmax(mutualComp)]


    