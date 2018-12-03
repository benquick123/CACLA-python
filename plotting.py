from cacla import Cacla
from arm import ArmController
import vrep
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    object_positions = pickle.load(open("object_locations.pickle", "rb"))[:-1]
    cacla_model = pickle.load(open("Saved_models/model_object_3_degrees.pickle", "rb"))
    arm = ArmController(clientID, joint_restrictions=[[-170, 170], [-135, 135], [-135, 135], [-90, 90], [-180, 180]])

    x_pos = []
    y_pos = []
    z_pos = []
    x_arm_pos = []
    y_arm_pos = []
    z_arm_pos = []
    for position, reached in object_positions:
        arm.reset_arm_position()
        if reached == 0:
            print("position:", position)
            x_pos.append(position[0])
            y_pos.append(position[1])
            z_pos.append(position[2])
            state = [[list(position) + [0.0] * 3]]

            _, a = cacla_model.predict(state, move=True)

            arm_pos = arm.get_tip_position()
            print("distance:", arm.get_distance())
            x_arm_pos.append(arm_pos[0])
            y_arm_pos.append(arm_pos[1])
            z_arm_pos.append(arm_pos[2])

    plot3D(x_pos, y_pos, z_pos, x_arm_pos, y_arm_pos, z_arm_pos)
