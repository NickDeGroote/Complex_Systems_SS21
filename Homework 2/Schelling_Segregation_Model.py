#####################################
# Team: Nicholas DeGroote, Lynn Pickering, Vita Borovyk, and Owen Traubert
#
# To run this code:
# At the bottom of the code, after the " if __name__ == "__main__": " statment
# - Change the parameters you desire to change such as size of board
# - Change the policy that you want to run
# - change the parameters that affect that policy
# - run the code, results are a plot of the mean happiness with standard deviation over the number of simulations run

import numpy as np
import copy
import matplotlib.pyplot as plt
import random


class SchellingSegregationModel:

    def __init__(
            self,
            k: int,
            simulation_environment_width: int,
            simulation_environment_height: int,
            population_density: float,
            epochs: int,
            q: int = 100,
            relocation_type: str = 'random',
            number_friends: int = 2,
            p: int = 3,
            max_distance: int = 10

    ):
        self.k = k
        self.simulation_environment_width = simulation_environment_width
        self.simulation_environment_height = simulation_environment_height
        self.population_density = population_density
        self.epochs = epochs
        self.environment = []
        self.q = q
        self.friends = []
        self.number_friends = number_friends
        self.p = p
        self.relocation_type = relocation_type
        self.max_dist = max_distance
        self.priority1 = []
        self.priority2 = []

    def create_environment(self) -> None:
        environment = np.random.choice([0, 1, 2],
                                       self.simulation_environment_width * self.simulation_environment_height,
                                       p=[1 - self.population_density, self.population_density / 2,
                                          self.population_density / 2])
        self.environment = environment.reshape(self.simulation_environment_width, self.simulation_environment_height)
        self.initial_enve = copy.deepcopy(self.environment)


    def create_priorities(self, pr1, pr2) -> None:
        self.priority1 = []
        self.priority2 = []
        for j in range(self.simulation_environment_height):
            row1=[]
            row2=[]
            for i in range(self.simulation_environment_width):
                if (i < self.simulation_environment_width//2)  and (j< self.simulation_environment_height//2):
                    row1.append(pr1[0])
                    row2.append(pr2[0])
                elif (i >= self.simulation_environment_width//2)  and (j< self.simulation_environment_height//2):
                    row1.append(pr1[1])
                    row2.append(pr2[1])
                elif (i < self.simulation_environment_width//2)  and (j >= self.simulation_environment_height//2):
                    row1.append(pr1[2])
                    row2.append(pr2[2])
                elif (i >= self.simulation_environment_width//2)  and (j>= self.simulation_environment_height//2):
                    row1.append(pr1[3])
                    row2.append(pr2[3])
            self.priority1.append(row1)
            self.priority2.append(row2)


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

        # happy_level = 0
        # # increase happy level for each neighbor of same type
        # for neighbor in neighbors_coord:
        #     # Had to reorder this array because I messed up order in neighbors_coord
        #     [j, i] = neighbor
        #     if self.environment[int(i), int(j)] == cell_type:
        #         happy_level += 1

        return happy_level

    def create_friends(self) -> None:
        # set all friends to zero
        self.friends = np.zeros(
            [self.simulation_environment_width * self.simulation_environment_height, self.number_friends], dtype=int)

        # line up all the cells
        environment_friends = self.initial_enve.reshape(
            self.simulation_environment_width * self.simulation_environment_height, 1)

        # check if the cell is occupied and that friend is different from the current cell
        for i in range(self.simulation_environment_width * self.simulation_environment_height):
            if (environment_friends[i] != 0):  # check that there is someone there
                for j in range(self.number_friends):  # choose a random cell number for a potential friend
                    friend = np.random.randint(0,
                                               self.simulation_environment_width * self.simulation_environment_height)
                    friendsAlready = False
                    for l in range(j):  # check if this cell is already our friend
                        if self.friends[i][l] == friend:
                            friendsAlready = True
                    # if friend is self or empty cell or a friend already choose another one
                    while (friend == i) or (environment_friends[friend] == 0) or (friendsAlready == True):
                        friend = np.random.randint(0,
                                                   self.simulation_environment_width * self.simulation_environment_height)
                        friendsAlready = False
                        for l in range(j):
                            if self.friends[i][l] == friend:
                                friendsAlready = True
                    self.friends[i][j] = friend

    def relocation_policy_random(self, current_agent_row, current_agent_col):
        z = 3
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
                return [rand_j, rand_i]
            else:
                checked_happy_levels.append([rand_j, rand_i, happy_level])

            checked += 1
        best_option = np.where(checked_happy_levels == max(checked_happy_levels[-1]))[0][0]

        return checked_happy_levels[best_option][0:2]

    def relocation_policy_social(self, cell_j, cell_i):
        new_location = [cell_j, cell_i]
        cell_type = self.environment[cell_j, cell_i]
        available_happy_places = []

        D1index = self.simulation_environment_width * cell_j + cell_i  # position in the linear stretched representation

        for friend in self.friends[D1index]:
            friend_j = friend // self.simulation_environment_width  # friend's location on the lattice
            friend_i = friend % self.simulation_environment_width

            for n in range(-(self.p // 2), self.p // 2 + 1):
                for m in range(-(self.p // 2), self.p // 2 + 1):
                    if (self.check_happiness((friend_j + n) % self.simulation_environment_width,
                                             (friend_i + m) % self.simulation_environment_height, cell_type) > 2) and (
                            self.environment[(friend_j + n) % self.simulation_environment_width, (
                                                                                                         friend_i + m) % self.simulation_environment_height] == 0):
                        available_happy_places.append([(friend_j + n) % self.simulation_environment_width,
                                                       (friend_i + m) % self.simulation_environment_height])

        if available_happy_places != []:
            new_location = random.choice(available_happy_places)

        return new_location

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

    def relocation_policy_priorities(self, cell_j, cell_i, priority1, priority2):
        new_location = [cell_j, cell_i]
        cell_type = self.environment[cell_j, cell_i]

        for best_place in range(4):
            if cell_type == 1:
                best_location = np.where(priority1==best_place)
                available_locations = np.where((self.environment == 0) & (self.priority1 == priority1[best_location]))
            elif cell_type == 2:
                best_location = np.where(priority2==best_place)
                available_locations = np.where((self.environment == 0) & (self.priority2 == priority2[best_location]))
            [available_locations_j, available_locations_i] = available_locations

            while len(available_locations_i) != 0:
                # get random location
                # TODO: need to seed this??
                random_location = np.random.randint(len(available_locations_i))
                rand_i = available_locations_i[random_location]  # column
                rand_j = available_locations_j[random_location]  # row
                # remove this location from lists
                available_locations_i = np.delete(available_locations_i, random_location)
                available_locations_j = np.delete(available_locations_j, random_location)
                cell_type = self.environment[cell_j, cell_i]
                happy_level = self.check_happiness(rand_j, rand_i, cell_type)
                if happy_level >= self.k:
                    return [rand_j, rand_i]

        return new_location


    def run_sim(self):
        self.create_environment()
        self.create_friends()

        priority1 = np.random.permutation([0, 1, 2, 3])
        priority2 = np.random.permutation([0, 1, 2, 3])

        # to use the same priority regions for both populations
        # priority2 = priority1

        self.create_priorities(priority1, priority2)

        happiness_at_epochs = []
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
                if self.relocation_type == "random":
                    if current_happy_level < 3:
                        [move_to_row, move_to_col] = self.relocation_policy_random(agent_row, agent_col)
                        self.environment[move_to_row, move_to_col] = self.environment[agent_row, agent_col]
                        self.environment[agent_row, agent_col] = 0

                elif self.relocation_type == "social":
                    [move_to_row, move_to_col] = self.relocation_policy_social(agent_row, agent_col)
                    if ([move_to_row, move_to_col] != [agent_row, agent_col]) and (current_happy_level < 3):
                        self.environment[move_to_row, move_to_col] = self.environment[agent_row, agent_col]
                        self.environment[agent_row, agent_col] = 0
                        self.friends[move_to_row * self.simulation_environment_width + move_to_col] = self.friends[
                            agent_row * self.simulation_environment_width + agent_col]
                        for people_friends in self.friends:
                            for friend in people_friends:
                                if friend == agent_row * self.simulation_environment_width + agent_col:
                                    ind = np.where(people_friends == friend)
                                    people_friends[ind] = move_to_row * self.simulation_environment_width + move_to_col
                        self.friends[agent_row * self.simulation_environment_width + agent_col] = [
                                                                                                      0] * self.number_friends

                elif self.relocation_type == "closest_distance":
                    [move_to_row, move_to_col, distance] = self.relocation_policy_closest(agent_row, agent_col)
                    self.environment[int(move_to_row), int(move_to_col)] = self.environment[agent_row, agent_col]

                    self.environment[agent_row, agent_col] = 0


                elif self.relocation_type == "priority_location":
                    if current_happy_level < 3:
                        [move_to_row, move_to_col] = self.relocation_policy_priorities(agent_row, agent_col, priority1, priority2)
                        if [move_to_row, move_to_col] != [agent_row, agent_col]:
                            self.environment[move_to_row, move_to_col] = self.environment[agent_row, agent_col]
                            self.environment[agent_row, agent_col] = 0

        return happiness_at_epochs, self.environment, self.initial_enve


if __name__ == "__main__":
    k = 3  # number of agents of own typ in neighborhood for agent j to be happy
    sim_env_width = 40
    sim_env_height = 40
    population_dens = .9  # how much of the environment is occupied by agents
    epochs = 20  # epochs to run simulation for
    cells_to_check_for_relocation = 100  # max cells to check for relocation, used in random and closest_distance

    # Relocation policies to choose from: random, social, closest_distance, priority_location
    relocation_policy = 'priority_location'

    # for the social policy
    number_of_friends = 5
    friend_neighborhood = 3

    # for the closest distance policy
    maximum_distance = 10

    model = SchellingSegregationModel(
        k=k,
        simulation_environment_width=sim_env_width,
        simulation_environment_height=sim_env_height,
        population_density=population_dens,
        epochs=epochs,
        q=cells_to_check_for_relocation,
        relocation_type=relocation_policy,
        number_friends=number_of_friends,
        p=friend_neighborhood,
        max_distance=maximum_distance)

    happiness_for_each_sim = np.array([])

    # how many simulations to run and average over
    sims_to_run = 3

    for sim in np.arange(0, sims_to_run):
        track_happiness, enve_final, enve_init = model.run_sim()
        happiness_for_each_sim = np.append([happiness_for_each_sim], np.array(track_happiness))
        print('Sim number: ', sim)

    happiness_for_each_sim = happiness_for_each_sim.reshape(sims_to_run, epochs)
    average_happiness = happiness_for_each_sim.sum(axis=0) / sims_to_run
    st_dev_happiness = np.std(happiness_for_each_sim, axis=0)

    epoch_array = np.arange(0, epochs)
    plt.figure(1)
    plt.errorbar(epoch_array, average_happiness, st_dev_happiness, label=(relocation_policy + " policy"))
    plt.title('Mean Happiness time-series with standard deviation')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Happiness')
    plt.legend()
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 2.0))
    plt.show()

    # View the last simulation's initital and final environments
    plt.figure(2)
    plt.imshow(enve_init, interpolation='none')
    plt.figure(3)
    plt.imshow(enve_final, interpolation='none')
    plt.show()
