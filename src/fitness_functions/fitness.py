from ConwayGOL import run_GOL

from fractal_fitness import defaultFractalFitness
import math
import statistics as stats
import numpy as np


def fractalFitness(initialConditions, timesteps):
    squaredDimensions = len(initialConditions)
    height = int(math.sqrt(squaredDimensions))
    width = height
    initialBoard= np.array(initialConditions)
    result2d, result1d = run_GOL(initialBoard, timesteps, width, height)
    fitnessList = defaultFractalFitness(result2d)
    meanFitness = stats.mean(fitnessList)
    return(meanFitness)

