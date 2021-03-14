import random
from typing import Callable, List

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
        migration_frequency: float = 1,
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
        self.migration_frequency = migration_frequency
        self.elitism_ratio = elitism_ratio
        self.all_populations = []

    def create_populations(self) -> None:
        """
        Creates the populations for the first generation of the MP GA
        :return: Population object containing Chromosomes
        """
        for i in range(0, self.num_populations):
            population = Population(
                fitness_function=self.fitness_function,
                population_size=self.population_size,
                crossover_probability=self.crossover_probability,
                mutation_probability=self.mutation_probability,
                elitism_ratio=self.elitism_ratio,
            )
            self.all_populations.append(population)
            population.create_chromosomes(self.input_data)

    def update_populations(self):
        """
        Performs the intra Population operations for the GA
        :return: None
        """
        for population in self.all_populations:
            population.update_to_next_generation()

    def migrate_to_new_population(self):
        """
        Takes a Chromosome from one Population and appends it to another
        :return: None
        """
        if random.random() < self.migration_probability:
            migrating_from_population, migrating_to_population = random.sample(
                self.all_populations, 2
            )
            migrating_chromosome = migrating_from_population.get_best_chromosome()
            migrating_to_population.accept_migrating_chromosome(migrating_chromosome)

    def run(self) -> None:
        """
        Call to run the MP GA
        :return: None
        """
        self.create_populations()
        for i in range(1, self.num_generations + 1):
            self.update_populations()
            # Perform migration every number of specified generations
            if i % self.migration_frequency == 0:
                self.migrate_to_new_population()

    def get_best_chromosomes(self) -> List[Chromosome]:
        """
        Returns the best Chromosome from each Population at the end of every generation
        :return: A Tuple of the best Chromosomes in each Population
        """
        best_chromosomes = []
        for population in self.all_populations:
            best_chromosomes.append(population.get_best_chromosome())
        return best_chromosomes
