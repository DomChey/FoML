# -*- coding: utf-8 -*-
import numpy as np

def absoluteEval(source, solved):
    # Returns the absolute score of a solved puzzle
    # input: source list of all pieces, solved list with indices and according pieces
#    solvedList = []
#    row = []
#    col = []
#    for p in solved:
#        row.append(p[0])
#        col.append(p[1])
#
#    coldiff = max(col) - min(col)
#    rowdiff = max(row) - min(row)
#    for p in solved:
#        xpos = p[0]-min(row)
#        ypos = p[1]-min(col)
#        solvedList.append((xpos, ypos, p[2]))
        
#    solvedList.sort(key=lambda p: p[0]*(coldiff+1)+p[1])
    solvedList = sorted(solved)
    correct = 0
    
    for i,p in enumerate(source):
        if np.array_equal(p, solvedList[i][2]):
            correct = correct+1
            
    return correct/len(source)


def relativeEval(solved):
    # input: solved list with indices and according pieces
    # Output: Relative Evaluation score
    
    length = len(solved)
    
    # Iterate through all pieces and find their neighbors as predicted
    for p1 in solved:
        row1 = p1[0]
        col1 = p1[1]
        
        for p2 in solved:
            row2 = p2[0]
            col2 = p2[1]
            if col2 == col1+1 and row2 == row1:
                p1[2].NeighborRight = p2[2].data
            elif col2 == col1 - 1 and row2 == row2:
                p1[2].NeighborLeft = p2[2].data
            elif col2 == col1 and row2 == row1 + 1:
                p1[2].NeighborDown = p2[2].data
            elif col2 == col1 and row2 == row1 - 1:
                p1[2].NeighborUp = p2[2].data
                
    # Count the correct neighbors
    correct = 0
    
    for p in solved:
        p = p[2] # extract piece from tupel
        
        if p.trueNeighborRight==[] or p.NeighborRight==[]:
            if (p.trueNeighborRight == p.NeighborRight):
                correct = correct + 1
        else:
            if (p.trueNeighborRight == p.NeighborRight).all():
                correct = correct + 1
            
        if p.trueNeighborLeft==[] or p.NeighborLeft==[]:
            if (p.trueNeighborLeft == p.NeighborLeft):
                correct = correct + 1
        else:
            if (p.trueNeighborLeft == p.NeighborLeft).all():
                correct = correct + 1
                
        if p.trueNeighborUp==[] or p.NeighborUp==[]:
            if (p.trueNeighborUp == p.NeighborUp):
                correct = correct + 1
        else:
            if (p.trueNeighborUp == p.NeighborUp).all():
                correct = correct + 1
                
        if p.trueNeighborDown==[] or p.NeighborDown==[]:
            if (p.trueNeighborDown == p.NeighborDown):
                correct = correct + 1
        else:
            if (p.trueNeighborDown == p.NeighborDown).all():
                correct = correct + 1
    
    return correct/(length*4)
    
                
    