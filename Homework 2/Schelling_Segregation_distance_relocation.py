import numpy as np
import copy
import matplotlib.pyplot as plt

class SchellingSegregationModel:

    def __init__(
            self,
            k: int,
            simulation_environment_width: int,
            simulation_environment_height: int,
            population_density: float,
            epochs: int,
            q: int = 100,
            max_dist: int = 40,
            number_friends: int = 2,
            p: int = 3
    ):
        self.p = p
        self.number_friends = number_friends
        self.k = k
        self.simulation_environment_width = simulation_environment_width
        self.simulation_environment_height = simulation_environment_height
        self.population_density = population_density
        self.epochs = epochs
        self.environment = []
        self.initial_enve = []
        self.q = q
        self.max_dist = max_dist

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
            if self.environment[int(i), int(j)] == cell_type:
                happy_level += 1
        return happy_level

    def relocation_policy_closest(self, current_agent_row, current_agent_col):
        # Find all available empty locations
        available_locations = np.where(self.environment == 0)
        [available_locations_j, available_locations_i] = available_locations

        distance_to_locations = ((available_locations_j - current_agent_row)**2
                                 + (available_locations_i - current_agent_col)**2) ** .5
        locations_w_distance = np.vstack((available_locations_j, available_locations_i, distance_to_locations))
        locations_sorted = locations_w_distance[:, locations_w_distance[-1].argsort()]

        idx_that_exceed_dist = np.where(locations_sorted[-1] > self.max_dist)
        # not to exceed maximum distance to move
        try:
            available_locations = locations_sorted[:, 0: idx_that_exceed_dist[0][0]]
        except IndexError:
            # none exceeded max distance, then go to max cells checked
            available_locations = locations_sorted[:, 0: self.q]

        [available_locations_j, available_locations_i, distances] = available_locations

        # backup list in case cell does not reach happiness level
        checked_happy_levels = []
        while len(available_locations_i) != 0:
            # get location
            coord_i = available_locations_i[0]  # column
            coord_j = available_locations_j[0]  # row
            distance = distances[0]
            # remove this location from lists
            available_locations_i = np.delete(available_locations_i, 0)
            available_locations_j = np.delete(available_locations_j, 0)
            distances = np.delete(distances, 0)

            cell_type = self.environment[current_agent_row, current_agent_col]
            happy_level = self.check_happiness(coord_i, coord_j, cell_type)
            if happy_level >= self.k:
                return [coord_j, coord_i, distance]
            else:
                checked_happy_levels.append([coord_j, coord_i, distance, happy_level])
        best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]
        return checked_happy_levels[best_option][0:3]

    def relocation_policy_random(self, current_agent_row, current_agent_col):
        available_locations = np.where(self.environment == 0)
        [available_locations_j, available_locations_i] = available_locations
        # counter for the number of new locations checked (no to exceed certain number)
        checked = 0
        # backup list in case cell does not reach happiness level
        checked_happy_levels = []
        while checked < self.q:
            # get random location
            # TODO: need to seed this??
            random_location = np.random.randint(len(available_locations_i))
            rand_i = available_locations_i[random_location]  # column
            rand_j = available_locations_j[random_location]  # row
            # remove this location from lists
            available_locations_i = np.delete(available_locations_i, random_location)
            available_locations_j = np.delete(available_locations_j, random_location)
            cell_type = self.environment[current_agent_row, current_agent_col]
            happy_level = self.check_happiness(rand_j, rand_i, cell_type)
            if happy_level >= self.k:
                distance = ((rand_j - current_agent_row)**2 + (rand_i - current_agent_col)**2)** .5
                return [rand_j, rand_i, distance]
            else:
                checked_happy_levels.append([rand_j, rand_i, happy_level])
        best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]
        distance = ((best_option[0] - current_agent_row) ** 2 + (best_option[1] - current_agent_col) ** 2) ** .5
        return checked_happy_levels[best_option][0:2], distance



    def run_sim(self):
        self.create_environment()
        happiness_at_epochs = []
        total_distance_moved = 0
        for epoch in np.arange(0, self.epochs):
            agent_locations = np.where(self.environment != 0)
            agent_rows, agent_columns = agent_locations
            number_agents = len(agent_rows)

            # get happiness at each epoch
            epoch_happiness = 0
            for row in np.arange(self.simulation_environment_height):
                for col in np.arange(self.simulation_environment_width):
                    agent_happy_level = self.check_happiness(row, col, self.environment[row, col])
                    if agent_happy_level >= 3:
                        epoch_happiness += 1
            happiness_at_epochs.append(epoch_happiness / number_agents)

            while len(agent_rows) != 0:
                random_agent = np.random.randint(len(agent_rows))
                agent_col = agent_columns[random_agent]  # column
                agent_row = agent_rows[random_agent]  # row
                # remove this agent from lists
                agent_rows = np.delete(agent_rows, random_agent)
                agent_columns = np.delete(agent_columns, random_agent)
                current_happy_level = self.check_happiness(agent_row, agent_col, self.environment[agent_row, agent_col])

                if current_happy_level < 3:
                    [move_to_row, move_to_col, distance] = self.relocation_policy_closest(agent_row, agent_col)
                    self.environment[int(move_to_row), int(move_to_col)] = self.environment[agent_row, agent_col]
                    self.environment[agent_row, agent_col] = 0
                    total_distance_moved += distance


        return happiness_at_epochs, self.environment, self.initial_enve, total_distance_moved

if __name__ == "__main__":

    k = 3  # number of agents of own type in neighborhood for agent j to be happy
    sim_env_width = 40
    sim_env_height = 40
    population_dens = .9  # how much of the environment is occupied by agents
    epochs = 101
    cells_to_check_for_relocation = 150
    max_distance = 1000
    model = SchellingSegregationModel(
        k=k,
        simulation_environment_width=sim_env_width,
        simulation_environment_height=sim_env_height,
        population_density=population_dens,
        epochs=epochs,
        q=cells_to_check_for_relocation,
        max_dist=max_distance)
    track_happiness, enve_final, enve_init, total_distance_moved = model.run_sim()

    # Plot initial environment and final environment for last simulation run
    plt.figure(1)
    plt.imshow(enve_init, interpolation='none')
    plt.figure(2)
    plt.imshow(enve_final, interpolation='none')
    print(track_happiness[-1])