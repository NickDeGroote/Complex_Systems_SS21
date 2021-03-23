import numpy as np


class SchellingSegregationModel:

    def __init__(
            self,
            k: int,
            simulation_environment_width: int,
            simulation_environment_height: int,
            population_density: float,
            epochs: int,
            relocation_policy,
            q: int = 100,

    ):
        self.k = k
        self.simulation_environment_width = simulation_environment_width
        self.simulation_environment_height = simulation_environment_height
        self.population_density = population_density
        self.epochs = epochs

    def create_environment(self):
        environment = np.random.choice([0, 1, 2],
                                       self.simulation_environment_width * self.simulation_environment_height,
                                       p=[1 - self.population_density, self.population_density / 2,
                                          self.population_density / 2])
        environment = environment.reshape(self.simulation_environment_width, self.simulation_environment_height)
        return environment


def relocation_policy_random():


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
        relocation_policy=relocation_policy_random,
        q=cells_to_check_for_relocation)


    enve = model.create_environment()
