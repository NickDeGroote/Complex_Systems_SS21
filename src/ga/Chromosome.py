import random
from typing import Callable, List

"""
Class for a Chromosome in the Genetic Algorithm
"""


class Chromosome:
    """
    Structure to hold the genes of an individual and its corresponding fitness
    """

    def __init__(self, genes: List[int]):
        self.genes = genes
        self.fitness = 0

    def mutate(self) -> None:
        """
        Mutates this Chromosome
        :return: None
        """
        mutation_index = random.randrange(len(self.genes))
        self.genes[mutation_index] = (0, 1)[self.genes[mutation_index] == 0]

    def calculate_fitness(self, fitness_function: Callable) -> None:
        """
        Calculates the fitness of this Chromosome
        :param fitness_function: Function evaluating fitness
        :return: Fitness of Chromosome
        """
        self.fitness = fitness_function(self.genes)
        return self.fitness
