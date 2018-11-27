import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_result(filenames):
    for filename in filenames:
        with open(filename, 'rb') as f:
            data = pickle.load(f)


        list_of_ntraj = [128,64,32,16,8,4]
        experiments = {128: np.zeros(50), 64:np.zeros(50), 32: np.zeros(50), 16: np.zeros(50), 8: np.zeros(50), 4: np.zeros(50)}
        for data_tuple in data:
            experiments[data_tuple[1]][data_tuple[0]] += data_tuple[2]
        experiments_means = np.zeros(6)
        experiments_vars = np.zeros(6)

        for key, value in experiments.items():
            print(key)
            print(value)
            experiments_means[list_of_ntraj.index(key)] = np.mean(value)
            experiments_vars[list_of_ntraj.index(key)] = np.std(value)
        labels=["Deep", "Linear"]
        cmap = ["g", "b", "r"]
        plt.plot([128,64,32,16,8,4], experiments_means, color = cmap[filenames.index(filename)], label=labels[filenames.index(filename)])
        plt.xticks([128,64,32,16,8,4])
       # plt.fill_between([128,64,32,16,8,4], experiments_means,
       #                  experiments_means+ experiments_vars * 1.96 / np.sqrt(20),
       #                   alpha=0.2, interpolate=True, facecolor = cmap[filenames.index(filename)])
       # plt.fill_between([128,64,32,16,8,4], experiments_means,
       #                  experiments_means- experiments_vars * 1.96 / np.sqrt(20),alpha=0.2, interpolate=True,facecolor = cmap[filenames.index(filename)])
    plt.legend()
    plt.xlabel("Number of Trajectories")
    plt.ylabel("Expected Value Difference")

    plt.savefig("figurenew.pdf")
    plt.show()

plot_result(["experiments_deep_binaryworld_h10_rep50.pkl"])



