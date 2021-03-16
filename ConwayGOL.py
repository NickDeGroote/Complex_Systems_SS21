import cellpylib as cpl
import numpy as np


def run_GOL(initial_condition: list, timesteps: list, board_width: int, board_height: int):
    """
    Runs a game of life simulation for the given time steps
    :param initial_condition: the gene from the GA
    :param timesteps: the initial time step, followed by the difference from initial to each next
    :param board_width: the width of the board
    :param board_height: the height of the board
    :return: Chromosome object for that individual
    """
    # shape from list to array
    initial_board = initial_condition.reshape((board_height, board_width))
    # Needs to be this format for the cpl package
    initial_board = np.array([initial_board])

    # evolve the cellular automaton for max given time steps
    result = cpl.evolve2d(initial_board, timesteps=(timesteps[-1]+1), neighbourhood='Moore',
                          apply_rule=cpl.game_of_life_rule)

    # return only the time steps requested
    # ca_at_time_steps = []
    ca_at_time_steps_1d = []
    for time_step in timesteps:
        # ca_at_time_steps.append(result[time_step])
        ca_at_time_steps_1d.append(result[time_step].reshape((1, board_height*board_width)))
    return ca_at_time_steps_1d


if __name__ == "__main__":
    init_conditions = np.array(np.random.randint(2, size=100))
    width = 50
    height = 2
    timesteps_to_run = [100, 120]

    result_2d, result_1d = run_GOL(init_conditions, timesteps_to_run, width, height)

