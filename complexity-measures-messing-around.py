#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:40:34 2021

@author: ulam
"""
import numpy as np
import math
import statistics

testCluster = [[1, 1, 0, 1],[0, 0, 0, 0],[0, 0 ,0, 0],[0, 0, 0, 1]]
testFractal = [[1,1,1,1,1,1,1,1],[1,0,1,0,1,0,1,0],[1,1,0,0,1,1,0,0],[1,0,0,0,1,0,0,0],[1,1,1,1,0,0,0,0],[1,0,1,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0]]




#given a 2d array and a size, this function will get the clusters and their membership

def getClusters(array, height, width):
    #an array to keep track of which cells have been visited
    visited = np.zeros((height, width))
   
    #two lists to keep track of the parents of clusters and cluster membership
    parentNodes = []
    clusterMembership = []
    childNodes = []
    
    #keeps track of the number of clusters
    numClusters = 0
    
    for i in range(height):
        for j in range(width):
            if(array[i][j] and not (visited[i][j])):
                #a stack to keep track of which cells need to be visited
                toVisit = []
                #add the first node to the list of parent nodes
                parentNodes.append([i, j])
                childNodes.append([])
                clusterMembership.append(1)
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
                                clusterMembership[numClusters] += 1
                                #add the cell to the list of child cells in the cluster
                                childNodes[numClusters].append([yCoord, xCoord])
                                #push the new cell onto the stack
                                toVisit.append([yCoord, xCoord])
                                visited[yCoord][xCoord] = 1
                                break
                        if(newCellFound):
                            break
                            
                    if(not newCellFound):
                        toVisit.pop()
    
    return(parentNodes, clusterMembership, childNodes, numClusters)
                               

def getClusterFitness(array, height, width, desiredClusters, desiredClusterSize, sizeConst, numConst):
    clusterInfo = getClusters(array, height, width)
    meanSize = statistics.mean(clusterInfo[1])
    numClusters = clusterInfo[3]
    
    sizeDiff = desiredClusterSize-meanSize
    #normalized fitness metric
    sizeScore = 1/(math.pow(np.e, (math.pow(sizeDiff, 2))))
    
    clusterDiff = desiredClusters-numClusters
    clusterScore = 1/(math.pow(np.e, (math.pow(clusterDiff, 2))))
    
    finalScore = (sizeConst*sizeScore + numConst*clusterScore)/(sizeConst+numConst)
    
    return(finalScore)
    

                        
#there are three clusters in a 4x4 arrary and 4 cells, TL, TR, BR corners filled with second to TL also filled
#output is right so this algorithm appears to be correct
getClusters(testCluster, 4, 4)                        
                     
#maybe a power of 3 would work better due to the moore neighborhood
def getFractalComplexity(array, height, width, R1, BASE):
    #R1 is the side length in number of cells of the least magnification
    #it is assumed that the greatest magnification will be the individual cell
    #ideally it is a power of BASE, and if it's not we pretend it is anyway
    #k is the number of different scales we can look at 
    k = int(round(math.log(R1, BASE)))
    print(k)

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
            #if you're not at the BASE/lowest magnification
            #create the tesselation for the next lowest magnification level
            levelH = int(scaleH)
            levelW = int(scaleW)
            scaleH = int(height/pow(BASE, i))
            scaleW = int(width/pow(BASE, i))
            if(not i==k):
   
                scaleArrays.append(np.zeros((scaleH, scaleW)))
                    
                #for your magnification level
                #this is in here so you don't have to check a million times whether this is the kth iteration
                for y in range(levelH):
                    for x in range(levelW):
                        #if this square is occupied at this magnification
                        if(scaleArrays[i][y][x]):
                            #increment the number of squares used at this level
                            numSquares[i] += 1
                            #set the corresponding square at the next lowest magnification to occupied
                            #doesn't matter how high this is as long as it's not 0, it's occupied
                            scaleArrays[i+1][int(math.floor(y/BASE))][int(math.floor(x/BASE))] += 1
            else:
                 for y in range(levelH):
                    for x in range(levelW):
                        if(scaleArrays[i][y][x]):
                            numSquares[i] += 1
                         
    dimensions = []
     
    for i in range(k):
        #calculate s epsilon for each differential magnification level
        sEp = numSquares[i]/numSquares[i+1]
        #get rid of the quotient by taking log BASE, log2(2) = 1
        #and magnification is BASEx
        
        dimension = math.log(sEp, BASE)
        
        dimensions.append(dimension)
        
    return(dimensions)
                 
    
def getFractalFitness(array, height, width, R1, base, minDim, maxDim):
    #this will make fitness close to 0 when it's one fractal dimesion away
    #from the goal
    scalingConstant = 4
    
    dimensions = getFractalComplexity(array, height, width, R1, base)
    meanDimension = statistics.mean(dimensions)
    
    #this allows for neutral space where fitness is 1 for a range
    diff = 0
    if(meanDimension<minDim):
        diff = meanDimension - minDim
    elif(meanDimension>maxDim):
        diff = meanDimension - maxDim
    
    fitness = math.pow(np.e, -scalingConstant*diff)
    
    return(fitness, meanDimension)
    
    
    
#testFractal is the sierpinski triangle rotated 30 degrees
#the fractal complexity of the sierpinski triangle is 1.58 so
#the algorithm appears to be correct  
print(getFractalFitness(testFractal, 8, 8, 4, 2, 1.5, 1.8))
                           
#getFractalComplexity(testFractal, 8, 8, 4, 2)   
getClusters(testFractal, 8,8)                            
                                
                                
                                
                                