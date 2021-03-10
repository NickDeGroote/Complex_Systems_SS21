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

    def calculate_fitness(self, fitness_function: Callable) -> float:
        """
        Calculates the fitness of this Chromosome
        :param fitness_function: Function evaluating fitness
        :return: Fitness of Chromosome
        """
