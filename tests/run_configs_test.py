import time
import pandas as pd
import h5py
import numpy as np

from src.ga.MP_Genetic_Algorithm import MPGeneticAlgorithm
from src.fitness_functions.sphere import sphere

"""
Test file for the Multi-Population Genetic Algorithm. A 50-dimensional 
Sphere Function is used with 5 populations to test the functionality.
"""

# Initialize seed data. This is currently unused...
seed = [50, 50, 50, 50, 50]

# Name this correctly for what you are testing
name_of_test = "Number of Populations 2"

# Whatever quantity you are testing - make it a list
param_change = [3, 4, 5]  # Number of populations

def_population_size = 50
def_generations = 100
def_crossover_probability = 0.8
def_mutation_probability = 0.05
def_migration_probability = 0.1
def_migration_frequency = 1
def_elitism_ratio = 0.02

track_run_time = []
track_final_fit_value = []
track_final_chromosome = []

# Must rename this to match what you are testing
for def_num_populations in param_change:

    # Start timer at current time
    start_time = time.time()

    # Create the MP GA object
    mp_ga = MPGeneticAlgorithm(
        input_data=seed,
        fitness_function=sphere,
        num_genes=50,
        num_populations=def_num_populations,
        population_size=def_population_size,
        generations=def_generations,
        crossover_probability=def_crossover_probability,
        mutation_probability=def_mutation_probability,
        migration_probability=def_migration_probability,
        migration_frequency=def_migration_frequency,
        elitism_ratio=def_elitism_ratio,
    )

    # Run the MP GA
    mp_ga.run()
    # Get the best Chromosome from the MP GA run
    best_chromosomes = mp_ga.get_best_chromosomes()

    # Print the execution time of the MP GA
    run_time = time.time() - start_time
    print("Run Time: %s seconds" % (run_time))

    # Print the best Chromosomes from each Population
    current_population_fits = []
    current_population_chromosomes = []
    for i in range(mp_ga.num_populations):
        current_population_fits.append(best_chromosomes[i].fitness)
        current_population_chromosomes.append(best_chromosomes[i].genes)

    # Track lists of the data
    track_run_time.append(run_time)
    track_final_fit_value.append(current_population_fits)
    track_final_chromosome.append(current_population_chromosomes)

df_final = pd.DataFrame(
    {
        name_of_test: param_change,
        "run time": track_run_time,
        "fit val": track_final_fit_value,
    }
)

df_final.to_excel((name_of_test + "_test_results.xlsx"))

# save chromosomes separately
with open(name_of_test + "_test_chroms.txt", "w") as filehandle:
    for listitem in track_final_chromosome:
        filehandle.write("%s\n" % listitem)
