from arm import ArmController
from cacla import Cacla
import vrep
import time
from log_to_file import LogToFile
import pickle
import numpy as np


def test():
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    arm = ArmController(clientID)

    cacla = pickle.load(open("model_object_test_multi_loc_v2.pickle", "rb"))
    cacla.arm.reset_arm_position()
    pos = cacla.arm.reset_object_position()
    state = [list(pos) + [0.0, 0.0, 0.0, 0.0, 0.0]]
    r, A = cacla.predict(np.array(state))
    return r, A


def train():
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    arm = ArmController(clientID)

    input_dim = 8
    output_dim = 5
    alpha = 0.001       # learning rate for actor
    beta = 0.01         # learning rate for critic
    gamma = 0.0         # discount factor
    exploration_probability = 0.4
    cacla = Cacla(arm, input_dim, output_dim, alpha, beta, gamma, exploration_probability)

    log = LogToFile()

    arm.reset_arm_position()
    arm.reset_object_position()

    n_epochs = 10000
    max_iter = 25
    learning_decay = 0.995
    exploration_factor = 1.0

    arm.train(cacla, n_epochs, max_iter, learning_decay, exploration_factor, log)

    time.sleep(0.1)
    vrep.simxFinish(clientID)


if __name__ == "__main__":
    train()

    exit()
