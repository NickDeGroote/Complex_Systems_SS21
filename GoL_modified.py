# Python code to implement Conway's Game Of Life
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


# setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]


testCluster = [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
testFractal = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
]


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
    print(parentNodes, clusterMembership, "\n")
    print(childNodes)


# there are three clusters in a 4x4 arrary and 4 cells, TL, TR, BR corners filled with second to TL also filled
# output is right so this algorithm appears to be correct
getClusters(testCluster, 4, 4)

# maybe a power of 3 would work better due to the moore neighborhood
def getFractalComplexity(array, height, width, R1, BASE):
    # R1 is the side length in number of cells of the least magnification
    # it is assumed that the greatest magnification will be the individual cell
    # ideally it is a power of BASE, and if it's not we pretend it is anyway
    # k is the number of different scales we can look at
    k = int(round(math.log(R1, BASE)))
    print(k)

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
        sEp = numSquares[i] / numSquares[i + 1]
        # get rid of the quotient by taking log BASE, log2(2) = 1
        # and magnification is BASEx

        dimension = math.log(sEp, BASE)

        dimensions.append(dimension)

    print(numSquares, "\n", dimensions)


# testFractal is the sierpinski triangle rotated 30 degrees
# the fractal complexity of the sierpinski triangle is 1.58 so
# the algorithm appears to be correct
getFractalComplexity(testFractal, 8, 8, 4, 2)


def randomGrid(N):

    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.4, 0.6]).reshape(N, N)


def addGlider(i, j, grid):

    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 255], [255, 0, 255], [0, 255, 255]])
    grid[i : i + 3, j : j + 3] = glider


def addGosperGliderGun(i, j, grid):

    """adds a Gosper Glider Gun with top left
       cell at (i, j)"""
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 255
    gun[6][1] = gun[6][2] = 255

    gun[3][13] = gun[3][14] = 255
    gun[4][12] = gun[4][16] = 255
    gun[5][11] = gun[5][17] = 255
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 255
    gun[7][11] = gun[7][17] = 255
    gun[8][12] = gun[8][16] = 255
    gun[9][13] = gun[9][14] = 255

    gun[1][25] = 255
    gun[2][23] = gun[2][25] = 255
    gun[3][21] = gun[3][22] = 255
    gun[4][21] = gun[4][22] = 255
    gun[5][21] = gun[5][22] = 255
    gun[6][23] = gun[6][25] = 255
    gun[7][25] = 255

    gun[3][35] = gun[3][36] = 255
    gun[4][35] = gun[4][36] = 255

    grid[i : i + 11, j : j + 38] = gun


def update(frameNum, img, grid, N):

    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):

            # compute 8-neghbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulaton takes place on a toroidal surface.
            total = int(
                (
                    grid[i, (j - 1) % N]
                    + grid[i, (j + 1) % N]
                    + grid[(i - 1) % N, j]
                    + grid[(i + 1) % N, j]
                    + grid[(i - 1) % N, (j - 1) % N]
                    + grid[(i - 1) % N, (j + 1) % N]
                    + grid[(i + 1) % N, (j - 1) % N]
                    + grid[(i + 1) % N, (j + 1) % N]
                )
                / ON
            )

            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 3) or (total > 5):
                    newGrid[i, j] = OFF
            else:
                if total == 4:
                    newGrid[i, j] = ON

    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    # getClusters(grid[:], N, N)
    getFractalComplexity(grid[:], N, N, 4, 2)
    return (img,)


# main() function
def main():

    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Runs Conway's Game of Life simulation."
    )

    # add arguments
    parser.add_argument("--grid-size", dest="N", required=False)
    parser.add_argument("--mov-file", dest="movfile", required=False)
    parser.add_argument("--interval", dest="interval", required=False)
    parser.add_argument("--glider", action="store_true", required=False)
    parser.add_argument("--gosper", action="store_true", required=False)
    args = parser.parse_args()

    # set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)

    # set animation update interval
    updateInterval = 50
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])

    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N * N).reshape(N, N)
        addGlider(1, 1, grid)
    elif args.gosper:
        grid = np.zeros(N * N).reshape(N, N)
        addGosperGliderGun(10, 10, grid)

    else:  # populate grid with random on/off -
        # more off than on
        grid = randomGrid(N)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation="nearest")
    ani = animation.FuncAnimation(
        fig,
        update,
        fargs=(img, grid, N),
        frames=10,
        interval=updateInterval,
        save_count=50,
    )

    # # of frames?
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=["-vcodec", "libx264"])

    plt.show()


# call main
if __name__ == "__main__":
    main()
