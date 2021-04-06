import os
import time
import inspect
import pandas as pd
from src.fitness_functions.fitness import fractalFitness, clusterDistanceFitness, clusterSizeFitness
from src.ga.MP_Genetic_Algorithm import MPGeneticAlgorithm

"""
Template for setting up Game of Life tests and saving results
"""

# Define Parameters for test
name_of_test = os.path.basename(__file__)[:-3] # Filename w/o .py extension
# Percentage of cells that will be alive when initializing populations
seed = [60, 60, 60]
# Range of iterations which will be used to evaluate fitness
start = 40
stop = 60
eval_range = range(start, stop)
# Callback for the fitness function being used
fitness_function = lambda ic: clusterDistanceFitness(ic, eval_range)
# Number of initial condition cells
k = 4
num_genes = (2**k) ** 2
# Number of populations being used in the MP GA
num_populations = len(seed)
# Number of Chromosomes initially in each Population
population_size = int(num_genes / 4)
# A migration event will be analyzed after this many generations
migration_frequency = 1
# Probability that a migration will occur when being analyzed
migration_probability = 0.1

# GA stochastic parameters
crossover_probability = 0.75
mutation_probability = 0.03
elitism_ratio = 0.05
generations = 100
# Start timer at current time
start_time = time.time()

# Create the MP GA object
mp_ga = MPGeneticAlgorithm(
    input_data=seed,
    fitness_function=fitness_function,
    num_genes=num_genes,
    num_populations=num_populations,
    population_size=population_size,
    generations=generations,
    crossover_probability=crossover_probability,
    mutation_probability=mutation_probability,
    migration_probability=migration_probability,
    migration_frequency=migration_frequency,
    elitism_ratio=elitism_ratio,
)

# Run the MP GA
mp_ga.run()
# Get the best Chromosome from the MP GA run
best_chromosomes = mp_ga.get_best_chromosomes()

# Print the execution time of the MP GA
run_time = time.time() - start_time
print("Run Time: %s seconds\n" % run_time)

# Print the best Chromosomes from each Population
for i in range(mp_ga.num_populations):
    print("Population {} best Chromosome - (Fitness, [Genes]):".format(i))
    print(best_chromosomes[i])

mp_ga.generate_plots()

# Print the best Chromosomes from each Population
current_population_fits = []
current_population_chromosomes = []
for i in range(mp_ga.num_populations):
    current_population_fits.append(best_chromosomes[i].fitness)
    current_population_chromosomes.append(best_chromosomes[i].genes)

df_final = pd.DataFrame(
    {
        "name of test": str(name_of_test),
        "run time": run_time,
        "fit val": current_population_fits,
        "seed": seed,
        "eval start": start,
        "eval stop": stop,
        "complexity type": inspect.getsourcelines(fitness_function)[0][0],
        "k": k,
        "number of genes": num_genes,
        "number of populations": num_populations,
        "migration frequency": migration_frequency,
        "migration probability": migration_probability,
        "crossover probability": crossover_probability,
        "mutation probability": mutation_probability,
        "elitism ratio": elitism_ratio,
        "number of generations": generations,
    }
)

df_final.to_excel(("test_logs/" + name_of_test + "_results.xlsx"))
print("\nSaved parameters to " + name_of_test + "_results.xlsx")

# save chromosomes separately
with open("test_logs/" + name_of_test + "_chroms.txt", "w") as filehandle:
    for listitem in current_population_chromosomes:
        filehandle.write("%s\n" % listitem)
print("Saved Chromosome to " + name_of_test + "_test_chroms.txt")
