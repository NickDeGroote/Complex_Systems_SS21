import cellpylib as cpl
import pandas as pd
import numpy as np
import ast
import h5py


def to_proper_list(inputlist):
    list = inputlist.to_list()
    converted = []
    for sublist in list:
        converted.append(ast.literal_eval(sublist))
    return converted


def get_best_chromosome_in_this_excel(fit, chrom):
    """
    Finds the best chromosome in the entire excel
    """
    index_best_row = fit.index(max(fit))
    index_best = fit[index_best_row].index(max(fit[index_best_row]))
    return np.array(chrom[index_best_row][index_best])


def get_specific_chromosome_in_this_excel(chroms, row_in_table, loc_in_row):
    """
    Finds the best chromosome in the entire excel
    """

    return np.array(chroms[row_in_table][loc_in_row])


test_name = "Number of Populations 2"

# define an empty list
chromosomes = []
# open file and read the content in a list
with open(test_name + "_test_chroms.txt", "r") as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        chromosomes.append(ast.literal_eval(currentPlace))


read_data = pd.read_excel(test_name + "_test_results.xlsx")
fitness = to_proper_list(read_data["fit val"])

# run_chromosome = get_best_chromosome_in_this_excel(fitness, chromosomes)
run_chromosome = np.array(np.random.randint(2, size=50 * 50))

# Board conditions
max_timestep = 60
board_width = 50
board_height = 50

# shape from list to array
initial_board = run_chromosome.reshape((board_height, board_width))
# Needs to be this format for the cpl package
cellular_automaton = np.array([initial_board])


# evolve the cellular automaton
cellular_automaton = cpl.evolve2d(
    cellular_automaton,
    timesteps=60,
    neighbourhood="Moore",
    apply_rule=cpl.game_of_life_rule,
)

cpl.plot2d_animate(cellular_automaton)
