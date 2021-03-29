import time

from src.fitness_functions.fitness import fractalFitness
from src.ga.MP_Genetic_Algorithm import MPGeneticAlgorithm

"""
Test file for the Multi-Population Genetic Algorithm. A 50-dimensional 
Sphere Function is used with 5 populations to test the functionality.
"""

# Values refer to the PERCENTAGE of a one in a Chromosome for each population
# IMPORTANT: Each value must be between 0 and 100
seed = [35, 45, 55]

# Start timer at current time
start_time = time.time()

# Create the MP GA object
mp_ga = MPGeneticAlgorithm(
    input_data=seed,
    fitness_function=lambda ic: fractalFitness(ic, range(40, 60)),
    num_genes=900,
    num_populations=3,
    population_size=10,
    generations=3,
    crossover_probability=0.8,
    mutation_probability=0.01,
    migration_probability=0.1,
    migration_frequency=1,
    elitism_ratio=0.05,
)

# Run the MP GA
mp_ga.run()
# Get the best Chromosome from the MP GA run
best_chromosomes = mp_ga.get_best_chromosomes()

# Print the execution time of the MP GA
print("Run Time: %s seconds\n" % (time.time() - start_time))

# Print the best Chromosomes from each Population
for i in range(mp_ga.num_populations):
    print("Population {} best Chromosome - (Fitness, [Genes]):".format(i))
    print(best_chromosomes[i])

#mp_ga.generate_plots()
