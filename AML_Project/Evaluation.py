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

