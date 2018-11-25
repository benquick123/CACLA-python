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

    cacla = pickle.load(open("model_object.pickle", "rb"))
    cacla.arm.reset_arm_position()
    pos = cacla.arm.reset_object_position()
    state = [list(pos) + [0.0, 0.0, 0.0, 0.0, 0.0]]
    r, A = cacla.predict(np.array(state))
    return r, A


def train():
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    joint_restrictions = None #[[-170, 170], [-135, 135], [-135, 135], [-90, 90], [-180, 180]]
    arm = ArmController(clientID, joint_restrictions)

    input_dim = 8
    output_dim = 5
    alpha = 0.005       # learning rate for actor
    beta = 0.01         # learning rate for critic
    gamma = 0.8         # discount factor
    exploration_factor = 0.5
    cacla = Cacla(arm, input_dim, output_dim, alpha, beta, gamma, exploration_factor)

    log = LogToFile()
    log.log("alpha: " + str(alpha) + ", beta: " + str(beta) + ", gamma: " + str(gamma) + ", exploration factor: " + str(exploration_factor))

    arm.reset_arm_position()
    arm.reset_object_position()

    n_epochs = 625
    max_iter = 40
    learning_decay = 0.998

    log.log("n_epochs: " + str(n_epochs) + ", max_iter: " + str(max_iter) + ", learning decay: " + str(learning_decay))
    log.log("comments: changed reward function to give rewards on interval [-1, 0].")
    log.write()

    arm.train(cacla, n_epochs, max_iter, learning_decay, log)

    time.sleep(0.1)
    vrep.simxFinish(clientID)


if __name__ == "__main__":
    """for i in range(20):
        r, _ = test()
        print(r)
    exit()"""
    train()

    exit()
