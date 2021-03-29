import numpy as np
import math
import statistics


# given a 2d array and a size, this function will get the clusters and their membership


def getClusters(array, height, width):
    # an array to keep track of which cells have been visited
    visited = np.zeros((height, width))

    # two lists to keep track of the parents of clusters and cluster membership
    parentNodes = []
    clusterMembership = []
    childNodes = []

    # keeps track of the number of clusters
    numClusters = 0

    for i in range(height):
        for j in range(width):
            if array[i][j] and not (visited[i][j]):
                # a stack to keep track of which cells need to be visited
                toVisit = []
                # add the first node to the list of parent nodes
                parentNodes.append([i, j])

                childNodes.append([])
                clusterMembership.append(1)
                toVisit.append([i, j])
                visited[i][j] = 1

                # perform a depth-first traversal of neighbors of the parent node
                while toVisit:
                    # get coordinates of node at the top of the stack
                    y = toVisit[-1][0]
                    x = toVisit[-1][1]

                    # two lists to hold the coordinates in which to check for neighbors
                    yCoords = []
                    xCoords = []

                    # determine coordinates to check
                    # NOTE: could make this worse on memory but better on computing time
                    # by putting newCellFound outside the while loop
                    # to let the program know whether it needs to generate all this
                    # and just saving it the first time in the stack structure
                    if y == 0:
                        yCoords.append(y)
                        yCoords.append(y + 1)
                    elif y == height - 1:
                        yCoords.append(y - 1)
                        yCoords.append(y)
                    else:
                        yCoords.append(y - 1)
                        yCoords.append(y)
                        yCoords.append(y + 1)
                    if x == 0:
                        xCoords.append(x)
                        xCoords.append(x + 1)
                    elif x == width - 1:
                        xCoords.append(x - 1)
                        xCoords.append(x)
                    else:
                        xCoords.append(x - 1)
                        xCoords.append(x)
                        xCoords.append(x + 1)

                    # keeps track if a new cell has been found
                    newCellFound = False

                    for yCoord in yCoords:
                        for xCoord in xCoords:
                            # for an unvisited, occupied cell in the cluster
                            if array[yCoord][xCoord] and not visited[yCoord][xCoord]:
                                newCellFound = True
                                # increment the number of cells in this cluster
                                clusterMembership[numClusters] += 1
                                # add the cell to the list of child cells in the cluster
                                childNodes[numClusters].append([yCoord, xCoord])
                                # push the new cell onto the stack
                                toVisit.append([yCoord, xCoord])
                                visited[yCoord][xCoord] = 1
                                break
                        if newCellFound:
                            break

                    if not newCellFound:
                        toVisit.pop()
                numClusters += 1

    return (parentNodes, clusterMembership, childNodes, numClusters)


def getClusterCenters(array, height, width):
    clusterInfo = getClusters(array, height, width)
    parentNodes = clusterInfo[0]
    clusterMembership = clusterInfo[1]
    childNodes = clusterInfo[2]
    numClusters = clusterInfo[3]
    clusterCenters = []
    for i in range(numClusters):
        xSum = 0
        ySum = 0
        ySum += parentNodes[i][0]
        xSum += parentNodes[i][1]
        for point in childNodes[i]:
            ySum += point[0]
            xSum += point[1]
        xMean = int(round(xSum / clusterMembership[i]))
        yMean = int(round(ySum / clusterMembership[i]))
        clusterCenters.append([yMean, xMean])
    return (clusterCenters, clusterInfo)


def clusterDistanceFitness(array, height, width, desiredClusterSize):
    clusterInfo = getClusterCenters(array, height, width)
    clusterCenters = clusterInfo[0]
    clusterMembership = clusterInfo[1][1]
    numClusters = clusterInfo[1][3]
    centerH = height / 2
    centerW = width / 2
    sumDistance = 0

    for i in range(numClusters):
        if clusterMembership[i] >= desiredClusterSize:
            distance = math.hypot(
                clusterCenters[i][0] - centerH, clusterCenters[i][1] - centerW
            )
            sumDistance += distance

    return sumDistance


def defaultClusterDistanceFitness(timestepBoards):
    defaultDesiredClusterSize = 6

    fitnessList = []
    for board in timestepBoards:
        stepBoard = np.array(board)
        boardShape = np.shape(stepBoard)
        height = boardShape[0]
        width = boardShape[1]
        fitnessList.append(
            clusterDistanceFitness(stepBoard, height, width, defaultDesiredClusterSize)
        )

    return fitnessList

def getMaxClusterSizeFitness(array, height, width):
    clusterSizes = getClusters(array, height, width)[1]
    return(max(clusterSizes))

def maxClusterSizeFitness(timestepBoards):
    fitnessList = []
    for board in timestepBoards:
        stepBoard = np.array(board)
        boardShape = np.shape(stepBoard)
        height = boardShape[0]
        width = boardShape[1]
        fitnessList.append(getMaxClusterSizeFitness(stepBoard, height, width))
    return fitnessList

