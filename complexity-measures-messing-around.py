#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:40:34 2021

@author: ulam
"""
import numpy as np

#given a 2d array and a size, this function will compute number of clusters and average cluster size

def getClusters(array, height, width):
    #an array to keep track of which cells have been visited
    visited = np.zeros(height, width)
   
    #two lists to keep track of the parents of clusters and cluster membership
    parentNodes = [[],[],[]]
    #keeps track of the number of clusters
    numClusters = 0
    
    for i in range(height):
        for j in range(width):
            if(array[i][j] and not (visited[i][j])):
                #a stack to keep track of which cells need to be visited
                toVisit = []
                #add the first node to the list of parent nodes
                parentNodes[0].append([i, j])
                parentNodes[1][numClusters] += 1
                toVisit.append([i, j])
                visited[i][j] = 1
                
                #perform a depth-first traversal of neighbors of the parent node
                while(toVisit):
                    #get coordinates of node at the top of the stack
                    y = toVisit[-1][0]
                    x = toVisit[-1][1]
                
                    #two lists to hold the coordinates in which to check for neighbors
                    yCoords = []
                    xCoords = []
                
                    #determine coordinates to check      
                    #NOTE: could make this worse on memory but better on computing time
                    #by putting newCellFound outside the while loop
                    #to let the program know whether it needs to generate all this
                    #and just saving it the first time in the stack structure 
                    if(y == 0):
                        yCoords.append(y)
                        yCoords.append(y+1)
                    elif(y == height-1):
                        yCoords.append(y-1)
                        yCoords.append(y)
                    else:
                        yCoords.append(y-1)
                        yCoords.append(y)
                        yCoords.append(y+1)
                    if(x == 0):
                        xCoords.append(x)
                        xCoords.append(x+1)
                    elif(x == width - 1):
                        xCoords.append(x-1)
                        xCoords.append(x)
                    else:
                        xCoords.append(x-1)
                        xCoords.append(x)
                        xCoords.append(x+1)
                    
                    #keeps track if a new cell has been found
                    newCellFound = False
                    
                    for yCoord in yCoords:
                        for xCoord in xCoords:
                            #for an unvisited, occupied cell in the cluster
                            if(array[yCoord][xCoord] and not visited[yCoord][xCoord]):
                                newCellFound = True
                                #increment the number of cells in this cluster
                                parentNodes[numClusters][1] += 1
                                #add the cell to the list of child cells in the cluster
                                parentNodes[numClusters][2].append([yCoord, xCoord])
                                #push the new cell onto the stack
                                toVisit.append([yCoord, xCoord])
                                visited[yCoord][xCoord] = 1
                                break
                        if(newCellFound):
                            break
                            
                    if(not newCellFound):
                        toVisit.pop()
                                
                     
#maybe make a constant so it's not always a power of 2
#maybe a power of 3 would work better due to the moore neighborhood
def getFractalComplexity(array, height, width, R1):
    #R1 is the side length in number of cells of the least magnification
    #it is assumed that the greatest magnification will be the individual cell
    #ideally it is a power of 2, and if it's not we pretend it is anyway
    #k is the number of different scales we can look at 
    k = (int)(round(log(R1, 2)))

    #create a list in which to hold all the tesselated arrays and the number of squares it takes
    #to cover the live cells of the automaton at each level
    scaleArrays = []
    
    scaleH = height
    scaleW = width
    
    #start it off with the greatest magnification and work up
    scaleArrays.append(np.array(array))
    #create empty list to hold number of sqares at each magnification level
    numSquares = []    
    
    #for each magnification level
    for i in range(k+1):
            #start with zero squares counted at this magnification
            numSquares.append(0)
            #if you're not at the base/lowest magnification
            #create the tesselation for the next lowest magnification level
            scaleH = height/pow(2, i)
            scaleW = width/pow(2, i)
            if(not i==k):
            scaleH = height/pow(2, i)
            scaleW = width/pow(2, i)        
            scaleArrays.append[np.zeros(scaleH, scaleW), 0]
                scaleArrays.append(np.zeros(scaleH, scaleW))
                    
                #for your magnification level
                #this is in here so you don't have to check a million times whether this is the kth iteration
                for y in range(2 * scaleH):
                    for x in range(2 * scaleW):
                        #if this square is occupied at this magnification
                        if(scaleArrays[i][y][x]):
                            #increment the number of squares used at this level
                            numSquares[i] += 1
                            #set the corresponding square at the next lowest magnification to occupied
                            #doesn't matter how high this is as long as it's not 0, it's occupied
                            scaleArrays[i+1][(int)floor(y/2)][(int)floor(x/2)] += 1
            else:
                 for y in range(2 * scaleH):
                    for x in range(2 * scaleW):
                        if(scaleArrays[i][y][x]):
                            numSquares[i] += 1
                         
     dimensions = []
     
     for i in range(k):
         #calculate s epsilon for each differential magnification level
         sEp = numSquares[i]/numSquares[i+1]
         #get rid of the quotient by taking log base 2, log2(2) = 1
         #and magnification is 2x
         dimension = log(sEp, 2)
         dimensions.append(dimension)
         
            
                                
                                
                                
                                
                                
                                