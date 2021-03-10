from typing import Callable, List

from src.ga.Chromosome import Chromosome

"""
Class for a Population in the MP GA
"""


class Population:
    """
    Structure to hold the chromosomes for a population and the population fitness
    """

    def __init__(
        self,
        population_size: int = 50,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        elitism_ratio: float = 0.02,
    ):
        self.size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_ratio = elitism_ratio
        self.chromosomes = None
        self.population_fitness = 0
        self.crossover_function = self.single_point_crossover

    def create_chromosomes(self, seed: list) -> Chromosome:
        """
        Creates a chromosome from the data in seed.
        :param seed: Data needed to create the chromosome
        :return: Chromosome object for that individual
        """

    def single_point_crossover(
        self,
        parent_1: List[int],
        parent_2: List[int],
    ):
        """
        Performs the cross over operation on two parent Chromosomes to produce a child
        Chromosome
        :param parent_1: First parent breeding
        :param parent_2: Second parent breeding
        :return: The two child Chromosomes
        """

    def roulette_selection(self) -> Chromosome:
        """
        Select a Chromosome from this population using the Roulette Wheel method
        :return: The selected Chromosome
        """

    def tournament_selection(self) -> Chromosome:
        """
        Select a Chromosome from this population using a tournament selection
        :return: The selected Chromosome
        """

    def crossover_chromosomes(self):
        """
        Performs cross over on the Chromosomes in this Population
        :return:
        """

    def mutate_chromosomes(self) -> None:
        """
        Performs mutation on the Chromosomes in this Population
        :return: None
        """

    def calc_chromosome_fitnesses(self, fitness_function: Callable) -> None:
        """
        Calculates the fitness of all chromosomes in this Population
        :param fitness_function: Function evaluating fitness
        :return: None
        """

    def rank_chromosomes(self) -> None:
        """
        Ranks the Chromosomes in this Population by their respective fitness values
        :return: None
        """

    def calc_population_fitness(self) -> float:
        """
        Calculates the fitness of this population. Note that population fitness refers
        to the fitness of the population as a whole, not the fitness of its Chromosomes
        :return: The fitness of this population
        """

    def update_to_next_generation(self) -> None:
        """
        Performs selection, crossover, and mutation on this population to move to the
        next generation
        :return: None
        """

    def get_best_chromosome(self) -> Chromosome:
        """
        :return: The best Chromosome in this population
        """
