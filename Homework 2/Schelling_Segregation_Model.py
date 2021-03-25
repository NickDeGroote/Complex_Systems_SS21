import numpy as np
import copy

class SchellingSegregationModel:

    def __init__(
            self,
            k: int,
            simulation_environment_width: int,
            simulation_environment_height: int,
            population_density: float,
            epochs: int,
            q: int = 100,

    ):
        self.k = k
        self.simulation_environment_width = simulation_environment_width
        self.simulation_environment_height = simulation_environment_height
        self.population_density = population_density
        self.epochs = epochs
        self.environment = []
        self.q = q

    def create_environment(self) -> None:
        environment = np.random.choice([0, 1, 2],
                                       self.simulation_environment_width * self.simulation_environment_height,
                                       p=[1 - self.population_density, self.population_density / 2,
                                          self.population_density / 2])
        self.environment = environment.reshape(self.simulation_environment_width, self.simulation_environment_height)
        self.initial_enve = copy.deepcopy(self.environment)

    def check_happiness(self, cell_j, cell_i, cell_type):
        if cell_type == 0:
            return 0
        # first arge in array[_, _ ] is row, then col
        i_left = cell_i - 1
        i_right = cell_i + 1
        j_down = cell_j + 1
        j_up = cell_j - 1

        # implement wraparound environment
        if cell_i == 0:
            i_left = self.simulation_environment_width - 1
        elif cell_i == self.simulation_environment_width - 1:
            i_right = 0
        if cell_j == 0:
            j_up = self.simulation_environment_height - 1
        elif cell_j == self.simulation_environment_height - 1:
            j_down = 0

        # Build array of neighbor coordinates to check
        neighbors_coord = [[i_left, cell_j], [i_right, cell_j], [cell_i, j_up], [cell_i, j_down],
                           [i_left, j_up], [i_right, j_up], [i_left, j_down], [i_right, j_down]]
        happy_level = 0
        # increase happy level for each neighbor of same type
        for neighbor in neighbors_coord:
            # Had to reorder this array because I messed up order in neighbors_coord
            [j, i] = neighbor
            if self.environment[i, j] == cell_type:
                happy_level += 1

        return happy_level

    def relocation_policy_random(self, current_agent_row, current_agent_col):
        z = 3
        available_locations = np.where(self.environment == 0)
        # TODO: correct indexing???
        [available_locations_j, available_locations_i] = available_locations

        # counter for the number of new locations checked (no to exceed certain number)
        checked = 0
        # backup list in case cell does not reach happiness level
        checked_happy_levels = []

        while checked < self.q:
            # get random location
            # TODO: need to seed this??
            random_location = np.random.randint(len(available_locations_i))
            rand_i = available_locations_i[random_location] # column
            rand_j = available_locations_j[random_location] # row

            # remove this location from lists
            available_locations_i = np.delete(available_locations_i, random_location)
            available_locations_j = np.delete(available_locations_j, random_location)

            cell_type = self.environment[current_agent_row, current_agent_col]
            happy_level = self.check_happiness(rand_j, rand_i, cell_type)
            if happy_level >= self.k:
                return [rand_j, rand_i]
            else:
                checked_happy_levels.append([rand_j, rand_i, happy_level])

        best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]

        return checked_happy_levels[best_option][0:2]

    def run_sim(self):
        self.create_environment()
        happiness_at_epochs = []
        for epoch in np.arange(0, self.epochs):
            agent_locations = np.where(self.environment != 0)
            agent_rows, agent_columns = agent_locations

            while len(agent_rows) != 0:
                random_agent = np.random.randint(len(agent_rows))
                agent_col = agent_columns[random_agent]  # column
                agent_row = agent_rows[random_agent]  # row
                # remove this agent from lists
                agent_rows = np.delete(agent_rows, random_agent)
                agent_columns = np.delete(agent_columns, random_agent)

                current_happy_level = self.check_happiness(agent_row, agent_col, self.environment[agent_row, agent_col])
                if current_happy_level < 3:
                    [move_to_row, move_to_col] = self.relocation_policy_random(agent_row, agent_col)
                    self.environment[move_to_row, move_to_col] = self.environment[agent_row, agent_col]
                    self.environment[agent_row, agent_col] = 0

            epoch_happiness = 0
            for row in np.arange(self.simulation_environment_height):
                for col in np.arange(self.simulation_environment_width):
                    agent_happy_level = self.check_happiness(row, col, self.environment[row, col])
                    if agent_happy_level >= 3:
                        epoch_happiness += 1
            happiness_at_epochs.append(epoch_happiness)

        return happiness_at_epochs, self.environment, self.initial_enve

if __name__ == "__main__":
    k = 3  # number of agents of own typ in neighborhood for agent j to be happy
    sim_env_width = 40
    sim_env_height = 40
    population_dens = .9  # how much of the environment is occupied by agents
    epochs = 20
    cells_to_check_for_relocation = 100
    model = SchellingSegregationModel(
        k=k,
        simulation_environment_width=sim_env_width,
        simulation_environment_height=sim_env_height,
        population_density=population_dens,
        epochs=epochs,
        q=cells_to_check_for_relocation)

    track_happiness, enve_final, enve_init = model.run_sim()
