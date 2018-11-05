from arm import ArmController
from cacla import Cacla
import vrep
import time
import pickle
import numpy as np

if __name__ == "__main__":
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    arm = ArmController(clientID)
    """
    st0 = arm.reset_arm_position()
    st1 = arm.reset_object_position()
    print(st0, st1)
    st = np.array([st1 + list(st0)])
    cacla = pickle.load(open("model_object.pickle", "rb"))
    cacla.predict()
    exit()"""

    input_dim = 8
    output_dim = 5
    alpha = 0.001                                                                # learning rate for actor
    beta = 0.001                                                                  # learning rate for critic
    gamma = 0.9                                                                  # discount factor
    exploration_probability = 0.2
    # cacla = Cacla(arm, input_dim, output_dim, alpha, beta, gamma, exploration_probability)
    cacla = pickle.load(open("model_object.pickle", "rb"))
    cacla.arm.reset_arm_position()
    cacla.predict(np.array([[0, 0.25, 0.0250, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    time.sleep(30)

    arm.reset_arm_position()

    n_epochs = 10000
    max_iter = 100
    exploration_factor = 1.0

    arm.train(cacla, n_epochs, max_iter, exploration_factor)

    time.sleep(0.1)
    vrep.simxFinish(clientID)
    exit()
