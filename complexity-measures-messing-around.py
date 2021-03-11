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
                                
                                
                                
                                
                                