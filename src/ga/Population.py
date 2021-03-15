import random
from operator import attrgetter
from typing import Callable, Tuple

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
        fitness_function: Callable,
        population_size: int = 50,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        elitism_ratio: float = 0.02,
        selection_type: str = "tournament",
        crossover_type: str = "single_point",
    ):
        self.fitness_function = fitness_function
        self.size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_ratio = elitism_ratio
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.chromosomes = []
        self.has_ranked_chromosomes = False
        self.population_fitness = 0

    def create_chromosomes(self, seed: list) -> None:
        """
        Creates a chromosome from the data in seed.
        :param seed: Data needed to create the chromosome
        :return: Chromosome object for that individual
        """
        # TODO: For initial conditions, seed contains a list of weights for the percentage of living / dead cells
        for _ in range(self.size):
            # Random list of zeros and ones
            genes = [random.randint(0, 1) for _ in range(len(seed))]
            chromosome = Chromosome(genes=genes)
            self.chromosomes.append(chromosome)

    @staticmethod
    def single_point_crossover(parent_1: Chromosome, parent_2: Chromosome):
        """
        Performs the cross over operation on two parent Chromosomes to produce a child
        Chromosome
        :param parent_1: First parent breeding
        :param parent_2: Second parent breeding
        :return: The two child Chromosomes
        """
        index = random.randrange(1, len(parent_1.genes))
        child_1_genes = parent_1.genes[:index] + parent_2.genes[index:]
        child_2_genes = parent_2.genes[:index] + parent_1.genes[index:]
        child_1 = Chromosome(genes=child_1_genes)
        child_2 = Chromosome(genes=child_2_genes)
        return child_1, child_2

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
        # Pick 2 random Chromosomes
        members = random.sample(self.chromosomes, 2)
        # Sort the 2 selected Chromosomes with the highest fitness first
        members.sort(key=attrgetter("fitness"), reverse=True)
        # Return the fitter of the 2 Chromosomes
        return members[0]

    def crossover_chromosomes(
        self, parent_1, parent_2
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Performs cross over on the Chromosomes in this Population
        :return:
        """
        # Select the cross over method from the specified cross over type string
        if self.crossover_type == "single_point":
            crossover = self.single_point_crossover
        else:
            # Default to single point cross over
            crossover = self.single_point_crossover

        # Initialize children to be clones of parents. Otherwise could set to None...
        child_1, child_2 = parent_1, parent_2
        # Perform cross over if random number is less than cross over probability
        if random.random() < self.crossover_probability:
            child_1, child_2 = crossover(parent_1, parent_2)
        return child_1, child_2

    def select_parents(self) -> Tuple[Chromosome, Chromosome]:
        # Select the selection method from the specified selection type string
        if self.selection_type == "tournament":
            selection = self.tournament_selection
        elif self.selection_type == "roulette":
            selection = self.roulette_selection
        else:
            selection = "tournament"
        # Gets two parents via the selection method above
        parent_1 = selection()
        parent_2 = selection()
        return parent_1, parent_2

    def mutate_chromosome(self, chromosome: Chromosome) -> None:
        """
        Performs mutation on the Chromosomes in this Population
        :return: None
        """
        # Mutates Chromosome if random number is less than mutation probability
        if random.random() < self.mutation_probability:
            chromosome.mutate()

    def calc_chromosome_fitnesses(self) -> None:
        """
        Calculates the fitness of all chromosomes in this Population
        :return: None
        """
        # Calculates fitness for each Chromosome in the population
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness(self.fitness_function)

    def rank_chromosomes(self) -> None:
        """
        Ranks the Chromosomes in this Population by their respective fitness values
        :return: None
        """
        # Ranks Chromosomes by their "fitness" attribute
        self.chromosomes.sort(key=attrgetter("fitness"), reverse=True)
        self.has_ranked_chromosomes = True

    def calc_population_fitness(self) -> float:
        """
        Calculates the fitness of this population. Note that population fitness refers
        to the fitness of the population as a whole, not the fitness of its Chromosomes
        :return: The fitness of this population
        """
        total_pop_fitness = 0
        for chromosome in self.chromosomes:
            total_pop_fitness += chromosome.fitness
        self.population_fitness = total_pop_fitness / self.size
        return self.population_fitness

    def create_next_generation(self) -> None:
        """
        Performs selection, crossover, and mutation on this population to for the
        next generation of this population
        :return: None
        """
        next_generation = []

        # Move the N fittest chromosomes to the next generation based on elitism
        if self.elitism_ratio:
            # Round number of elites to nearest integer
            num_elite = int(round(self.size * self.elitism_ratio))
            elite_chromosomes = self.chromosomes[0:num_elite]
            next_generation[0:num_elite] = elite_chromosomes

        while len(next_generation) < self.size:

            parent_1, parent_2 = self.select_parents()

            child_1, child_2 = self.crossover_chromosomes(parent_1, parent_2)
            # Perform mutation if random number is less than mutation probability
            self.mutate_chromosome(child_1)
            self.mutate_chromosome(child_2)

            next_generation.append(child_1)
            if len(next_generation) < self.size:
                next_generation.append(child_2)

        self.has_ranked_chromosomes = False
        self.chromosomes = next_generation

    def update_to_next_generation(self) -> None:
        """
        1. Create a new population
        2. Calculate Chromosome fitnesses
        3. Rank the Chromosomes
        :return: None
        """
        self.create_next_generation()
        self.calc_chromosome_fitnesses()
        self.rank_chromosomes()

    def accept_migrating_chromosome(self, chromosome: Chromosome) -> None:
        """
        Adds a migrating Chromosome to this population. Adjusts the population size
        accordingly
        :return: The best Chromosome in this population
        """
        self.chromosomes.append(chromosome)
        self.size = len(self.chromosomes)

    def get_best_chromosome(self) -> Chromosome:
        """
        :return: The best Chromosome in this population
        """
        # Make sure that the Chromosomes are ranked if we call this
        if not self.has_ranked_chromosomes:
            self.rank_chromosomes()
        return self.chromosomes[0]
