import random
from operator import attrgetter
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

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
        num_genes: int,
        num_populations: int = 5,
        population_size: int = 50,
        generations: int = 100,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.01,
        migration_probability: float = 0.1,
        migration_frequency: float = 1,
        elitism_ratio: float = 0.02,
    ):
        self.input_data = input_data
        self.fitness_function = fitness_function
        self.num_genes = num_genes
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
                num_genes=self.num_genes,
                population_size=self.population_size,
                crossover_probability=self.crossover_probability,
                mutation_probability=self.mutation_probability,
                elitism_ratio=self.elitism_ratio,
            )
            self.all_populations.append(population)
            population.create_chromosomes(self.input_data, i)

    def update_populations(self) -> None:
        """
        Performs the intra Population operations for the GA
        :return: None
        """
        for population in self.all_populations:
            population.log_population_attributes()
            population.update_to_next_generation()

    def rank_populations(self) -> None:
        """
        Ranks the Chromosomes in this Population by their respective fitness values
        :return: None
        """
        # Ranks Chromosomes by their "fitness" attribute
        self.all_populations.sort(key=attrgetter("population_fitness"), reverse=True)

    def change_population_sizes(self) -> None:
        """
        Changes the sizes of Populations based on average Population fitness
        :return: None
        """
        self.rank_populations()
        increasing_population, decreasing_population = (
            self.tournament_population_selection()
        )
        if increasing_population is not None:
            # Minimum population size of 5
            if decreasing_population.size > 5:
                increasing_population.size += 1
                decreasing_population.size -= 1

    def tournament_population_selection(self) -> Tuple[Population, Population]:
        """
        Gets a Population based on the weight of Population average fitness
        :return: None
        """
        # Pick 2 random Chromosomes
        members = random.sample(self.all_populations, 2)
        # Sort the 2 selected Chromosomes with the highest fitness first
        members.sort(key=attrgetter("population_fitness"), reverse=True)
        # Return the fitter of the 2 Chromosomes
        return members[0], members[1]

    def migrate_to_new_population(self) -> None:
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
            self.change_population_sizes()
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

    def generate_plots(self) -> None:
        """
        Generates three plots over the lifecycle of the MP GA:
        1. Population Best Chromosome Fitness
        2. Population Average Fitness Over Generations
        3. Population Size Over Generations
        :return: None
        """
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        for population in self.all_populations:
            best_chrom_fitnesses = []
            generations = []
            legends = []
            for generation in range(0, self.num_generations):
                generations.append(generation)
                best_chrom_fitnesses.append(
                    population.best_chromosome_log[generation].fitness
                )
                legends.append("Population {}".format(generation))

            ax1.plot(generations, best_chrom_fitnesses)
            # The first average fitness is zero, ignore it
            ax2.plot(generations[1:], population.population_fitness_log[1:])
            ax3.plot(generations, population.population_size_log)
            ax1.title.set_text("Population Best Chromosome Fitness Over Generations")
            ax2.title.set_text("Population Average Fitness Over Generations")
            ax3.title.set_text("Population Size Over Generations")
            ax1.set_xlabel("Generation")
            ax2.set_xlabel("Generation")
            ax3.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax2.set_ylabel("Average Fitness")
            ax3.set_ylabel("# of Chromosomes")
            ax1.legend(legends)
            ax2.legend(legends)
            ax3.legend(legends)
        plt.show()
