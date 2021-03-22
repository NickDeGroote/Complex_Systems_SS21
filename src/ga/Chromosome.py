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

    def mutate(self, index) -> None:
        """
        Mutates this Chromosome
        :return: None
        """
        # Swaps the value of a random gene in this Chromosome
        self.genes[index] = (0, 1)[self.genes[index] == 0]

    def calculate_fitness(self, fitness_function: Callable) -> None:
        """
        Calculates the fitness of this Chromosome
        :param fitness_function: Function evaluating fitness
        :return: Fitness of Chromosome
        """
        self.fitness = fitness_function(self.genes)
        return self.fitness

    def __repr__(self):
        """
        Return Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))
