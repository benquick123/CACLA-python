import numpy as np
import vrep
import keras

from arm import ArmController
from keras.models import Sequential
from keras.layers import Dense


class Cacla:
    def __init__(self, arm, input_dim, output_dim, n_actor_neurons, n_critic_neurons, alpha, exploration_probability):
        self.arm = arm
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.exploration_probability = exploration_probability

        self.actor = self._create_actor(input_dim, output_dim, n_actor_neurons, alpha)
        self.critic = self._create_critic(input_dim, 1, n_critic_neurons, alpha)

    def fit(self, iter_n):
        exploration_probability = self.exploration_probability * (1 - iter_n / n_iterations)

        pass

    def _create_actor(self, input_dim, output_dim, n_neurons, learning_rate):
        model = Sequential()
        model.add(Dense(n_neurons, input_dim=input_dim, activation="sigmoid"))
        model.add(Dense(output_dim, activation='sigmoid'))

        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model

    def _create_critic(self, input_dim, output_dim, n_neurons, learning_rate):
        model = Sequential()
        model.add(Dense(n_neurons, input_dim=input_dim, activation="sigmoid"))
        model.add(Dense(output_dim))

        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model


if __name__ == '__main__':
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID == -1:
        print("Connection to V-REP not successful.")
        exit()

    parameters = [                  # 3,            # number of dimensions of robot world (1D / 2D / 3D)
                                    # [1],          # degrees of freedom of the arm
                  5,                # ,100],        # number of neurons in critic
                  6,                # ,150],        # number of neurons in actor
                  0.05,             # ,0.005],      # earning rate for actor ANNs
                  0.09,             # ,0.01],       # learning rate for critic ANN
                                    # 'sigmoid',        # activation function in first layer of actor and critic ANNs (sigmoid / tanh / softplus...)
                  'new',            # type of exploration - probabilities for exploration (old / new)
                  0.9,              # initial exploration probability
                                    # 'SGD',            # optimization algorithms for actor and critic (SGD / ...add other)
                  'quadratic',      # type of reward (linear / quadratic)
                  'state',          # type of output of actor (state / action)
                  0.95,             # discount factor
                  0.8,              # exploration factor - how large are explorations
                  50000,            # number of epochs
                  'continuous',     # type of cues (continuous / linguistic)
                  1,                # number of succesful reaches until change of the position of the target
                  'change',         # type of action - as change of state or as new state
                  1]                # scaling of action

    learner = Cacla(clientID, *parameters)
    learner.train()

