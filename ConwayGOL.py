import cellpylib as cpl
import numpy as np


def game_of_life_rule(neighbourhood, c, t):
    """
    param die_under_pop: any live cell with fewer than 'a' live neighbors dies (under population)
    param die_over_pop: any live cell with more than 'b' live neighbors dies (over population)
    param live_reproduction: any dead cell with 'c' neighbors becomes live (reproduction)
    """
    die_under_pop = 2
    die_over_pop = 3
    live_reproduction = 3

    center_cell = neighbourhood[1][1]
    total = np.sum(neighbourhood)
    if center_cell == 1:
        if total - 1 < die_under_pop:
            return 0  # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        if total - 1 == die_under_pop or total - 1 == die_over_pop:
            return 1  # Any live cell with two or three live neighbours lives on to the next generation.
        if total - 1 > die_over_pop:
            return 0  # Any live cell with more than three live neighbours dies, as if by overpopulation.
    else:
        if total == live_reproduction:
            return 1  # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        else:
            return 0


def run_GOL(
    initial_condition: np.ndarray, timesteps: list, board_width: int, board_height: int
):
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
    result = cpl.evolve2d(
        initial_board,
        timesteps=(timesteps[-1] + 1),
        neighbourhood="Moore",
        apply_rule=game_of_life_rule,
    )

    # return only the time steps requested
    # ca_at_time_steps = []
    ca_at_time_steps_1d = []
    for time_step in timesteps:
        # ca_at_time_steps.append(result[time_step])
        ca_at_time_steps_1d.append(
            result[time_step].reshape((1, board_height * board_width))
        )
    return ca_at_time_steps_1d


if __name__ == "__main__":
    init_conditions = np.array(np.random.randint(2, size=100))
    width = 50
    height = 2
    timesteps_to_run = [100, 120]


    result_1d = run_GOL(init_conditions, timesteps_to_run, width, height)
