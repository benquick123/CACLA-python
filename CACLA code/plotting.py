import vrep
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vrep_arm2 as arm
import numpy as np


def plot3D(x_pos, y_pos, z_pos, x_arm_pos, y_arm_pos, z_arm_pos):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(x_arm_pos)):
        ax.plot([x_pos[i], x_arm_pos[i]], [y_pos[i], y_arm_pos[i]], [z_pos[i], z_arm_pos[i]], "-", c="#00000033")
    ax.scatter(x_pos, y_pos, z_pos, c="#6699ff")
    ax.scatter(x_arm_pos, y_arm_pos, z_arm_pos, c="#ff6600")

    plt.show()


if __name__ == "__main__":
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    cacla = pickle.load(open("C:/Users/Jonathan/Documents/School/Project_Farkas/CACLA code/log-files/V-rep_AL5D/Jan-15_13.48.07_best_so_far/model.pickle", "rb"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_pos = []
    y_pos = []
    z_pos = []
    distances = []
    colors = []
    n = 300
    for i in range(n):
        observation = cacla.env.reset()
        info = {"distance": 1.0}
        done = False
        while not done:
            cacla.env.render()
            action = cacla.actor.predict(np.array([observation]))

            observation, reward, done, info = cacla.env.step(action[0])
        x_pos.append(observation[0])
        y_pos.append(observation[1])
        z_pos.append(observation[2])
        distances.append(max(1.0 - info["distance"], 0.0))
        colors.append("red" if info["distance"] < 0.01 else "blueviolet")

    distances = np.array(distances)
    distances = (distances - min(distances)) / (max(distances) - min(distances))
    for i in range(n):
        ax.scatter(x_pos[i], y_pos[i], z_pos[i], alpha=distances[i], c=colors[i])
    plt.show()
