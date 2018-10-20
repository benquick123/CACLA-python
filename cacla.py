import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense


class Cacla:
    def __init__(self, arm, input_dim, output_dim, n_actor_neurons, n_critic_neurons, alpha, gamma, exploration_probability):
        self.arm = arm
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.exploration_probability = exploration_probability

        self.actor = self._create_actor(input_dim, output_dim, n_actor_neurons, alpha)
        self.critic = self._create_critic(input_dim, 1, n_critic_neurons, alpha)

    def fit(self, state_vect_t0, exploration_factor):
        # for now exploration will be linear function
        _exploration_probability = self.exploration_probability * exploration_factor
        A_t0 = self.actor.predict(state_vect_t0, batch_size=1)
        A_t0 = np.array(A_t0).flatten()

        a = self._choose_action(A_t0, _exploration_probability)
        state_vect_t1 = np.reshape(np.append(state_vect_t0[0][:3], a), (1, -1))
        A_t1 = self.actor.predict(state_vect_t1, batch_size=1)

        self.arm.joints_move(A_t1)
        r_t1 = self.arm.get_distance()
        self.arm.joints_move(A_t0)

        V_t0 = self.critic.predict(state_vect_t0, batch_size=1)[0][0]
        V_t1 = self.critic.predict(state_vect_t1, batch_size=1)[0][0]

        delta = r_t1 + self.gamma * V_t1 - V_t0
        self.critic.fit(state_vect_t1, [[r_t1 + self.gamma * V_t1]], batch_size=1)

        if delta > 0:
            self.actor.fit(state_vect_t1, A_t1, batch_size=1)
            self.arm.joints_move(A_t1)

            if self.arm.get_distance() < 0.01:                  # if distance is smaller than 1 cm
                return 0, state_vect_t1                         # 0 = "done"
            return 1, state_vect_t1                             # 1 = "in progress"

        return 1, state_vect_t0                                 # 1 = "in progress"

    def fit_iter(self, state_vect, exploration_factor, max_iter):
        for i in range(max_iter):
            train_state, state_vect = self.fit(state_vect, exploration_factor)
            if train_state == 0:
                print("Reached in", i, "iterations.")
                return 0                                        # successful reach
        print("Reach unsuccessful.")
        return -1                                               # unsuccessful reach

    def get_reward(self, distance):
        max_distance = 2
        rd = 1 - 2 * (distance / max_distance)
        return -rd * np.abs(rd)

    @staticmethod
    def _choose_action(action, explore):
        e = [np.random.normal() * explore for i in range(len(action))]
        return action + e

    @staticmethod
    def _create_actor(input_dim, output_dim, n_neurons, learning_rate):
        model = Sequential()
        model.add(Dense(n_neurons, input_dim=input_dim, activation="sigmoid"))
        model.add(Dense(output_dim, activation='sigmoid'))

        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model

    @staticmethod
    def _create_critic(input_dim, output_dim, n_neurons, learning_rate):
        model = Sequential()
        model.add(Dense(n_neurons, input_dim=input_dim, activation="sigmoid"))
        model.add(Dense(output_dim))

        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model


if __name__ == '__main__':
    cacla = Cacla(None, 9, 6, 5, 5, 0.1, 0.5, 0.5)
    cacla.fit(np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]]), 1.0)
    exit()
