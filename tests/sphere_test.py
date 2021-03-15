import time

from src.ga.MP_Genetic_Algorithm import MPGeneticAlgorithm
from src.fitness_functions.sphere import sphere

"""
Test file for the Multi-Population Genetic Algorithm. A 50-dimensional 
Sphere Function is used with 5 populations to test the functionality.
"""

# Initialize seed data. This is currently unused...
seed = [0] * 50

# Start timer at current time
start_time = time.time()

# Create the MP GA object
mp_ga = MPGeneticAlgorithm(
    input_data=seed,
    fitness_function=sphere,
    num_populations=5,
    population_size=50,
    generations=100,
    crossover_probability=0.8,
    mutation_probability=0.05,
    migration_probability=0.1,
    migration_frequency=1,
    elitism_ratio=0.02,
)

# Run the MP GA
mp_ga.run()
# Get the best Chromosome from the MP GA run
best_chromosomes = mp_ga.get_best_chromosomes()

# Print the execution time of the MP GA
print("Run Time: %s seconds" % (time.time() - start_time))

# Print the best Chromosomes from each Population
for i in range(mp_ga.num_populations):
    print("Population {} best Chromosome - (Fitness, [Genes]):".format(i))
    print(best_chromosomes[i])
