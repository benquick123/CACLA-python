from arm import ArmController
from cacla import Cacla
import vrep
import time

if __name__ == "__main__":
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    arm = ArmController(clientID)

    input_dim = 9
    output_dim = 6
    n_neurons_actor = 100
    n_neurons_critic = 100
    alpha = 0.1                                                                 # learning rate for neural networks
    gamma = 0.5                                                                 # discount factor
    exploration_probability = 1.0
    cacla = Cacla(arm, input_dim, output_dim, n_neurons_actor, n_neurons_critic, alpha, gamma, exploration_probability)

    arm.reset_arm_position()

    n_iterations = 20000
    arm.train(cacla, n_iterations)

    time.sleep(0.1)
    vrep.simxFinish(clientID)
    exit()
