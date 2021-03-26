from ConwayGOL import run_GOL

from fractal_fitness import defaultFractalFitness
from cluster_fitness import defaultClusterDistanceFitness
import math
import statistics as stats
import numpy as np


def getGOLResults(initialConditions, timesteps):
    squaredDimensions = len(initialConditions)
    height = int(math.sqrt(squaredDimensions))
    width = height
    initialBoard= np.array(initialConditions)
    result2d, result1d = run_GOL(initialBoard, timesteps, width, height)
    return(result2d)

def fractalFitness(initialConditions, timesteps):
    result2d = getGOLResults(initialConditions, timesteps)
    fitnessList = defaultFractalFitness(result2d)
    meanFitness = stats.mean(fitnessList)
    return(meanFitness)

def clusterDistanceFitness(initialConditions, timesteps):
    result2d = getGOLResults(initialConditions, timesteps)
    fitnessList = defaultClusterDistanceFitness(result2d)
    print(fitnessList)
    meanFitness = stats.mean(fitnessList)
    return(meanFitness)

