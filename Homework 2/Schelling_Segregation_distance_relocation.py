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

        # increase happy level for each neighbor of same type
        happy_level = sum(int(self.environment[int(i), int(j)] == cell_type) for j, i in neighbors_coord)

        return happy_level

    def relocation_policy_closest(self, current_agent_row, current_agent_col):
        # Find all available empty locations
        available_locations = np.where(self.environment == 0)
        [available_locations_j, available_locations_i] = available_locations

        distance_to_locations = ((available_locations_j - current_agent_row) ** 2
                                 + (available_locations_i - current_agent_col) ** 2) ** .5
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
            happy_level = self.check_happiness(coord_j, coord_i, cell_type)
            # make sure it does not count itself as a neighbor when moving to a neighboring position!
            if distance == 1 or distance == 2 ** .5:
                happy_level -= 1
            if happy_level >= self.k:
                return [coord_j, coord_i, distance]
            else:
                checked_happy_levels.append([coord_j, coord_i, distance, happy_level])
        try:
            best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]
        except IndexError:
            # no locations to move to that are close enough
            return [current_agent_row, current_agent_col, 0]
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
            random_location = np.random.randint(len(available_locations_i))
            rand_i = available_locations_i[random_location]  # column
            rand_j = available_locations_j[random_location]  # row

            # remove this location from lists
            available_locations_i = np.delete(available_locations_i, random_location)
            available_locations_j = np.delete(available_locations_j, random_location)
            cell_type = self.environment[current_agent_row, current_agent_col]
            happy_level = self.check_happiness(rand_j, rand_i, cell_type)
            if happy_level >= self.k:
                distance = ((rand_j - current_agent_row) ** 2 + (rand_i - current_agent_col) ** 2) ** .5
                return [rand_j, rand_i, distance]
            else:
                checked_happy_levels.append([rand_j, rand_i, happy_level])
            checked += 1
        best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]
        distance = ((best_option[0] - current_agent_row) ** 2 + (best_option[1] - current_agent_col) ** 2) ** .5
        return checked_happy_levels[best_option][0:2], distance

    def run_sim(self):
        self.create_environment()
        happiness_at_epochs = []
        total_distance_moved_at_epochs = []
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
            total_distance_moved_at_epochs.append(total_distance_moved)

        return happiness_at_epochs, self.environment, self.initial_enve, total_distance_moved_at_epochs


if __name__ == "__main__":
    sims_to_run = 2

    k = 3  # number of agents of own type in neighborhood for agent j to be happy
    sim_env_width = 40
    sim_env_height = 40
    population_dens = .9  # how much of the environment is occupied by agents
    epochs = 21
    cells_to_check_for_relocation = 100
    max_distances = [3, 5, 10, 15]

    plot_all_sims_happy = np.array([])
    plot_all_sims_dist_moved = np.array([])
    for max_distance in max_distances:
        model = SchellingSegregationModel(
            k=k,
            simulation_environment_width=sim_env_width,
            simulation_environment_height=sim_env_height,
            population_density=population_dens,
            epochs=epochs,
            q=cells_to_check_for_relocation,
            max_dist=max_distance)

        happiness_for_each_sim = np.array([])
        dist_for_each_sim = np.array([])
        for sim in np.arange(0, sims_to_run):
            print(max_distance, sim)
            track_happiness, enve_final, enve_init, total_distance_moved = model.run_sim()
            happiness_for_each_sim = np.append([happiness_for_each_sim], np.array(track_happiness))
            dist_for_each_sim = np.append([dist_for_each_sim], np.array(total_distance_moved))

        happiness_for_each_sim = happiness_for_each_sim.reshape(sims_to_run, epochs)
        average_happiness = happiness_for_each_sim.sum(axis=0) / sims_to_run
        plot_all_sims_happy = np.append([plot_all_sims_happy], average_happiness)

        dist_for_each_sim = dist_for_each_sim.reshape(sims_to_run, epochs)
        average_distance = dist_for_each_sim.sum(axis=0) / sims_to_run
        plot_all_sims_dist_moved = np.append([plot_all_sims_dist_moved], average_distance)

    no_max_dist_hap = [0.78042175, 0.97024305, 0.99376897, 0.99814846, 0.99937418,
                        0.9996754, 0.99990706, 1., 1., 1.,
                        1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1.,
                        1.]
    no_max_dist_dist = [792.65052131, 1001.06837023, 1069.09083818, 1095.06819318,
                        1102.48779635, 1106.38966948, 1107.84041265, 1107.84041265,
                        1107.84041265, 1107.84041265, 1107.84041265, 1107.84041265,
                        1107.84041265, 1107.84041265, 1107.84041265, 1107.84041265,
                        1107.84041265, 1107.84041265, 1107.84041265, 1107.84041265,
                        1107.84041265]
    random_policy_comp_happy = [0.77757862, 0.96093731, 0.98961532, 0.9969216, 0.9990309,
                                0.99963403, 0.99995436, 0.9999774, 1., 1., 1., 1., 1., 1., 1.,
                                1., 1., 1., 1., 1., 1.]
    plot_all_sims_happy = plot_all_sims_happy.reshape(len(max_distances), epochs)
    sim3, sim5, sim10, sim15 = plot_all_sims_happy

    random_policy_comp_dist = [6270.87130973, 7674.08445364, 8066.77734958, 8190.0031193,
                               8237.62884937, 8251.87200157, 8252.87488627, 8255.08992616,
                               8255.08992616, 8255.08992616, 8255.08992616, 8255.08992616,
                               8255.08992616, 8255.08992616, 8255.08992616, 8255.08992616,
                               8255.08992616, 8255.08992616, 8255.08992616, 8255.08992616,
                               8255.08992616]
    plot_all_sims_dist_moved = plot_all_sims_dist_moved.reshape(len(max_distances), epochs)
    sim3d, sim5d, sim10d, sim15d = plot_all_sims_dist_moved

    epoch_array = np.arange(0, epochs)
    plt.figure(3)
    plt.plot(epoch_array, random_policy_comp_happy, label='Random Policy')
    plt.plot(epoch_array, sim3, label='Max Distance Checked = 3')
    plt.plot(epoch_array, sim5, label='Max Distance Checked = 5')
    plt.plot(epoch_array, sim10, label='Max Distance Checked = 10')
    plt.plot(epoch_array, sim15, label='Max Distance Checked = 15')
    plt.plot(epoch_array, no_max_dist_hap, label='No Max Distance')

    plt.title('Mean Happiness time-series')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Happiness')
    plt.legend()
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 2.0))

    plt.figure(4)
    plt.plot(epoch_array, random_policy_comp_dist, label='Random Policy')
    plt.plot(epoch_array, sim3d, label='Max Distance Checked = 3')
    plt.plot(epoch_array, sim5d, label='Max Distance Checked = 5')
    plt.plot(epoch_array, sim10d, label='Max Distance Checked = 10')
    plt.plot(epoch_array, sim15d, label='Max Distance Checked = 15')
    plt.plot(epoch_array, no_max_dist_dist, label='No Max Distance')
    plt.title('Mean Total Distance traveled over epochs time-series')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Distance traveled')
    plt.legend()
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 2.0))
    # # Plot initial environment and final environment for last simulation run
    # plt.figure(1)
    # plt.imshow(enve_init, interpolation='none')
    # plt.figure(2)
    # plt.imshow(enve_final, interpolation='none')
    # print(track_happiness[-1])
