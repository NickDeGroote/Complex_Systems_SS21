import math
import statistics
import cellpylib as cpl
import numpy as np

size = 10

def sphere(x: list) -> float:
    fitness = 100-sum(x)
    return fitness

def entropyValue(x: list) -> float:
    fitness = abs(cpl.shannon_entropy(x))
    return fitness


def getFractalComplexity(array2):
    height = size
    width = size
    R1 = 4
    BASE = 2
    array = np.reshape(array2, (height, width))
    # R1 is the side length in number of cells of the least magnification
    # it is assumed that the greatest magnification will be the individual cell
    # ideally it is a power of BASE, and if it's not we pretend it is anyway
    # k is the number of different scales we can look at
    k = int(round(math.log(R1, BASE)))
    #print(k)

    # create a list in which to hold all the tesselated arrays and the number of squares it takes
    # to cover the live cells of the automaton at each level
    scaleArrays = []

    scaleH = height
    scaleW = width

    # start it off with the greatest magnification and work up
    scaleArrays.append(np.array(array))
    # create empty list to hold number of squares at each magnification level
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
        if (not i == k):

            scaleArrays.append(np.zeros((scaleH, scaleW)))

            # for your magnification level
            # this is in here so you don't have to check a million times whether this is the kth iteration
            for y in range(levelH):
                for x in range(levelW):
                    # if this square is occupied at this magnification
                    if (scaleArrays[i][y][x]):
                        # increment the number of squares used at this level
                        numSquares[i] += 1
                        # set the corresponding square at the next lowest magnification to occupied
                        # doesn't matter how high this is as long as it's not 0, it's occupied
                        scaleArrays[i + 1][int(math.floor(y / BASE))][int(math.floor(x / BASE))] += 1
        else:
            for y in range(levelH):
                for x in range(levelW):
                    if (scaleArrays[i][y][x]):
                        numSquares[i] += 1

    dimensions = []

    for i in range(k):
        # calculate s epsilon for each differential magnification level
        sEp = numSquares[i] / numSquares[i + 1]
        # get rid of the quotient by taking log BASE, log2(2) = 1
        # and magnification is BASEx

        dimension = math.log(sEp, BASE)

        dimensions.append(dimension)

    return (dimension)


