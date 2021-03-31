import numpy as np
import matplotlib.pyplot as plt

from Schelling_Segregation import SchellingSegregationModel

if __name__ == "__main__":
    # How many sims to run of each
    sims_to_run = 30

    # Run the random policy
    k = 3  # number of agents of own type in neighborhood for agent j to be happy
    sim_env_width = 40
    sim_env_height = 40
    population_dens = .9  # how much of the environment is occupied by agents
    epochs = 21
    cells_to_check_for_relocation = 100
    model = SchellingSegregationModel(
        k=k,
        simulation_environment_width=sim_env_width,
        simulation_environment_height=sim_env_height,
        population_density=population_dens,
        epochs=epochs,
        q=cells_to_check_for_relocation)

    plot_all_sims = np.array([])
    plot_all_sims_st_dev = np.array([])
    happiness_for_each_sim = np.array([])

    for sim in np.arange(0, sims_to_run):
        # 1 = random, 2 = social
        relocation_type = 1
        track_happiness, enve_final, enve_init = model.run_sim(relocation_type)
        happiness_for_each_sim = np.append([happiness_for_each_sim], np.array(track_happiness))
        print('Sim random; number: ', sim)

    happiness_for_each_sim = happiness_for_each_sim.reshape(sims_to_run, epochs)
    average_happiness = happiness_for_each_sim.sum(axis=0) / sims_to_run
    st_dev_happiness = np.std(happiness_for_each_sim, axis=0)
    plot_all_sims = np.append([plot_all_sims], average_happiness)
    plot_all_sims_st_dev = np.append([plot_all_sims_st_dev], st_dev_happiness)

    # Run variations of the friends policy
    number_of_friends = [5, 10, 20]
    friend_neighborhoods = [3, 5]
    for num_friend in number_of_friends:
        for friend_neighborhood in friend_neighborhoods:
            model = SchellingSegregationModel(
                k=k,
                simulation_environment_width=sim_env_width,
                simulation_environment_height=sim_env_height,
                population_density=population_dens,
                epochs=epochs,
                number_friends=num_friend,
                p=friend_neighborhood)

            happiness_for_each_sim = np.array([])
            for sim in np.arange(0, sims_to_run):
                # 1 = random, 2 = social
                relocation_type = 2
                track_happiness, enve_final, enve_init = model.run_sim(relocation_type)
                happiness_for_each_sim = np.append([happiness_for_each_sim], np.array(track_happiness))
                print('Sim friends; number: ', sim)

            happiness_for_each_sim = happiness_for_each_sim.reshape(sims_to_run, epochs)
            average_happiness = happiness_for_each_sim.sum(axis=0) / sims_to_run
            st_dev_happiness = np.std(happiness_for_each_sim, axis=0)
            plot_all_sims = np.append([plot_all_sims], average_happiness)
            plot_all_sims_st_dev = np.append([plot_all_sims_st_dev], st_dev_happiness)
            print('Finished: n = ', num_friend, ' p = ', friend_neighborhood)

    # # Plot initial environment and final environment for last simulation run
    # plt.figure(1)
    # plt.imshow(enve_init, interpolation='none')
    # plt.figure(2)
    # plt.imshow(enve_final, interpolation='none')
    plot_all_sims = plot_all_sims.reshape(7, epochs)
    plot_all_sims_st_dev = plot_all_sims_st_dev.reshape(7, epochs)
    avg_hap_rand, avg_hap_fnd1, avg_hap_fnd2, avg_hap_fnd3, avg_hap_fnd4, avg_hap_fnd5, avg_hap_fnd6 = plot_all_sims
    st_dev_rand, st_dev_fnd1, st_dev_fnd2, st_dev_fnd3, st_dev_fnd4, st_dev_fnd5, st_dev_fnd6 = plot_all_sims_st_dev
    epoch_array = np.arange(0, epochs)
    plt.figure(3)
    plt.plot(epoch_array, avg_hap_rand, label='Random Policy')
    plt.plot(epoch_array, avg_hap_fnd1, label='Social Network Recommendation: n=5, p=3')
    plt.plot(epoch_array, avg_hap_fnd2, label='Social Network Recommendation: n=5, p=5')
    plt.plot(epoch_array, avg_hap_fnd3, label='Social Network Recommendation: n=10, p=3')
    plt.plot(epoch_array, avg_hap_fnd4, label='Social Network Recommendation: n=10, p=5')
    plt.plot(epoch_array, avg_hap_fnd5, label='Social Network Recommendation: n=15, p=3')
    plt.plot(epoch_array, avg_hap_fnd6, label='Social Network Recommendation: n=15, p=5')
    plt.title('Mean Happiness time-series')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Happiness')
    plt.legend()
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 2.0))

    # with error bars
    plt.figure(4)
    plt.errorbar(epoch_array, avg_hap_rand, st_dev_rand, label='Random Policy')
    plt.errorbar(epoch_array, avg_hap_fnd1, st_dev_fnd1, label='Social Network Recommendation: n=5, p=3')
    plt.errorbar(epoch_array, avg_hap_fnd2, st_dev_fnd2, label='Social Network Recommendation: n=10, p=3')
    plt.errorbar(epoch_array, avg_hap_fnd3, st_dev_fnd3, label='Social Network Recommendation: n=15, p=3')
    plt.errorbar(epoch_array, avg_hap_fnd4, st_dev_fnd4, label='Social Network Recommendation: n=5, p=5')
    plt.errorbar(epoch_array, avg_hap_fnd5, st_dev_fnd5, label='Social Network Recommendation: n=10, p=5')
    plt.errorbar(epoch_array, avg_hap_fnd6, st_dev_fnd6, label='Social Network Recommendation: n=15, p=5')
    plt.title('Mean Happiness time-series with Error Bars')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Happiness')
    plt.legend()
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 2.0))
    plt.show()
