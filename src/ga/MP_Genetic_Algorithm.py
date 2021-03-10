from typing import Callable, List, Tuple

from src.ga.Chromosome import Chromosome
from src.ga.Population import Population

"""
Class for Multi-population (MP) Genetic Algorithm (GA)
"""


class MPGeneticAlgorithm:
    def __init__(
        self,
        input_data: list,
        fitness_function: Callable,
        num_populations: int = 5,
        population_size: int = 50,
        generations: int = 100,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        migration_probability: float = 0.01,
        elitism_ratio: float = 0.02,
    ):
        self.input_data = input_data
        self.fitness_function = fitness_function
        self.num_populations = num_populations
        self.population_size = population_size
        self.num_generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.migration_probability = migration_probability
        self.elitism_ratio = elitism_ratio
        self.all_populations = None

    def create_populations(self) -> None:
        """
        Creates the populations for the first generation of the MP GA
        :return: Population object containing Chromosomes
        """

    def migrate_to_new_population(self):
        """
        Takes a Chromosome from one Population and appends it to another
        :return: None
        """

    def run(self) -> None:
        """
        Call to run the MP GA
        :return: None
        """

    def get_best_chromosomes(self) -> Tuple[Chromosome]:
        """
        Returns the best Chromosome from each Population at the end of every generation
        :return: A Tuple of the best Chromosomes in each Population
        """
