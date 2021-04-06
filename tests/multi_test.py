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
k = 2
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
generations = 2
# Start timer at current time
start_time = time.time()

param_change = [0, 10, 20]

track_run_time = []
track_final_fit_value = []
track_max_fitness = []
track_final_chromosome = []

for def_num_populations in param_change:
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

    #mp_ga.generate_plots()

    # Print the best Chromosomes from each Population
    current_population_fits = []
    current_population_chromosomes = []
    best_fitness = 0
    best_chrom = None
    for i in range(mp_ga.num_populations):
        current_population_fits.append(best_chromosomes[i].fitness)
        current_population_chromosomes.append(best_chromosomes[i].genes)
        if best_chromosomes[i].fitness > best_fitness:
            best_chrom = best_chromosomes[i]
            best_fitness = best_chromosomes[i].fitness

    # Track lists of the data
    track_run_time.append(run_time)
    track_final_fit_value.append(current_population_fits)
    track_max_fitness.append(max(current_population_fits))
    track_final_chromosome.append(best_chrom)

df_final = pd.DataFrame(
    {
        "name of test": [str(name_of_test)] * len(param_change),
        "run time": track_run_time,
        "fit vals": track_final_fit_value,
        "max fitness": track_max_fitness,
        "seed": [seed] * len(param_change),
        "eval start": start * len(param_change),
        "eval stop": stop * len(param_change),
        "complexity type": inspect.getsourcelines(fitness_function)[0][0] * len(param_change),
        "k": k * len(param_change),
        "number of genes": num_genes * len(param_change),
        "number of populations": num_populations * len(param_change),
        "migration frequency": migration_frequency * len(param_change),
        "migration probability": migration_probability * len(param_change),
        "crossover probability": crossover_probability * len(param_change),
        "mutation probability": mutation_probability * len(param_change),
        "elitism ratio": elitism_ratio * len(param_change),
        "number of generations": generations * len(param_change),
    }
)

df_final.to_excel(("test_logs/" + name_of_test + "_results.xlsx"))
print("\nSaved parameters to " + name_of_test + "_results.xlsx")

# save chromosomes separately
with open("test_logs/" + name_of_test + "_chroms.txt", "w") as filehandle:
    for listitem in track_final_chromosome:
        filehandle.write("%s\n" % listitem)
print("Saved Chromosome to " + name_of_test + "_test_chroms.txt")
