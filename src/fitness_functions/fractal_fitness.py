import numpy as np
import math
import statistics


def getFractalComplexity(array, height, width, R1, BASE):
    # R1 is the side length in number of cells of the least magnification
    # it is assumed that the greatest magnification will be the individual cell
    # ideally it is a power of BASE, and if it's not we pretend it is anyway
    # k is the number of different scales we can look at
    k = int(round(math.log(R1, BASE)))

    # create a list in which to hold all the tesselated arrays and the number of squares it takes
    # to cover the live cells of the automaton at each level
    scaleArrays = []

    scaleH = height
    scaleW = width

    # start it off with the greatest magnification and work up
    scaleArrays.append(np.array(array))
    # create empty list to hold number of sqares at each magnification level
    numSquares = []

    # for each magnification level
    for i in range(k + 1):
        # start with zero squares counted at this magnification
        numSquares.append(0)
        # if you're not at the BASE/lowest magnification
        # create the tesselation for the next lowest magnification level
        levelH = int(scaleH)
        levelW = int(scaleW)
        scaleH = int(height / pow(BASE, i))
        scaleW = int(width / pow(BASE, i))
        if not i == k:

            scaleArrays.append(np.zeros((scaleH, scaleW)))

            # for your magnification level
            # this is in here so you don't have to check a million times whether this is the kth iteration
            for y in range(levelH):
                for x in range(levelW):
                    # if this square is occupied at this magnification
                    if scaleArrays[i][y][x]:
                        # increment the number of squares used at this level
                        numSquares[i] += 1
                        # set the corresponding square at the next lowest magnification to occupied
                        # doesn't matter how high this is as long as it's not 0, it's occupied
                        scaleArrays[i + 1][int(math.floor(y / BASE))][
                            int(math.floor(x / BASE))
                        ] += 1
        else:
            for y in range(levelH):
                for x in range(levelW):
                    if scaleArrays[i][y][x]:
                        numSquares[i] += 1

    dimensions = []

    for i in range(k):
        # calculate s epsilon for each differential magnification level
        if numSquares[i] == 0:
            dimensions.append(0)
            continue
        sEp = numSquares[i] / numSquares[i + 1]
        # get rid of the quotient by taking log BASE, log2(2) = 1
        # and magnification is BASEx

        dimension = math.log(sEp, BASE)

        dimensions.append(dimension)

    return dimensions


# recommended values 1.58 to 1.9 for fractals
# default life rules has complexity ~1.0 to 1.1 when alive
# and ~0.5 - 0.7 when quiescent
def getFractalFitness(array, height, width, R1, base, minDim, maxDim):
    # this will make fitness close to 0 when it's one fractal dimesion away
    # from the goal
    scalingConstant = 4

    dimensions = getFractalComplexity(array, height, width, R1, base)
    meanDimension = statistics.mean(dimensions)

    # this allows for neutral space where fitness is 1 for a range
    diff = 0
    if meanDimension < minDim:
        diff = minDim - meanDimension
    elif meanDimension > maxDim:
        diff = meanDimension - maxDim

    fitness = math.pow(np.e, -scalingConstant * diff)

    return (fitness, meanDimension)


def defaultFractalFitness(timestepBoards):
    minDim = 1.58
    maxDim = 1.9

    fractalFitnessList = []
    for board in timestepBoards:
        stepBoard = np.array(board)
        boardShape = np.shape(stepBoard)
        height = boardShape[0]
        width = boardShape[1]

        R1 = None
        base = None
        if not (height % 2):
            base = 2
            R1 = 4
        elif not (height % 3):
            base = 3
            R1 = 9

        fractalInfo = getFractalFitness(
            stepBoard, height, width, R1, base, minDim, maxDim
        )
        fractalFitnessList.append(fractalInfo[0])

    return fractalFitnessList
