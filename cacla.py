import numpy as np
import keras
import time

from keras.models import Sequential
from keras.layers import Dense


class Cacla:
    def __init__(self, arm, input_dim, output_dim, alpha, beta, gamma, exploration_probability):
        self.arm = arm
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.exploration_probability = exploration_probability

        self.actor = self._create_actor(input_dim, output_dim, alpha)
        self.critic = self._create_critic(input_dim, 1, beta)

    def fit(self, state_vect_t0, exploration_factor):
        # for now exploration will be linear function
        _exploration_probability = self.exploration_probability * exploration_factor
        A_t0 = self.actor.predict(state_vect_t0, batch_size=1)
        A_t0 = np.array(A_t0).flatten()

        print("CURRENT STATE:", state_vect_t0[0][3:])
        a = self._choose_action(A_t0, _exploration_probability)
        state_vect_t1 = np.reshape(np.append(state_vect_t0[0][:3], a), (1, -1))

        print("default action:", A_t0)
        print("EXPLORING ACTION:", a)
        self.arm.joints_move(a)
        # above_floor = self.arm.above_floor()
        # print("ABOVE FLOOR:", above_floor)
        r_t1 = self.get_reward()
        # if not above_floor:
        #     r_t1 -= 1
        self.arm.joints_move(state_vect_t0[0][3:])

        V_t0 = self.critic.predict(state_vect_t0, batch_size=1)[0][0]
        V_t1 = self.critic.predict(state_vect_t1, batch_size=1)[0][0]

        print("REWARD:", r_t1)
        delta = r_t1 + self.gamma * V_t1 - V_t0
        print("DELTA:", delta)
        self.critic.fit(state_vect_t0, [[r_t1 + self.gamma * V_t1]], batch_size=1, verbose=0)

        if delta > 0:
            print("-----------------------------------------UPDATING ACTOR--------------------------------------------")
            self.actor.fit(state_vect_t0, np.array([a.tolist()]), batch_size=1, verbose=0)
            # print(state_vect_t1)
            # A_t1 = self.actor.predict(state_vect_t0, batch_size=1, verbose=0)
            # state_vect_t1 = np.array([np.append(state_vect_t0[0][:3], A_t1[0])])
            # print(state_vect_t1)
            print("PERFORMED ACTION:", a)
            self.arm.joints_move(a)

            if self.arm.get_distance() < 0.02:                  # if distance is smaller than 1 cm
                return 0, state_vect_t1                         # 0 = "done"
            return 1, state_vect_t1                             # 1 = "in progress"

        return 1, state_vect_t0                                 # 1 = "in progress"

    def predict(self, state_vect_t0, move=True):
        A = self.actor.predict(state_vect_t0)
        if move:
            self.arm.joints_move(A[0])

    def fit_iter(self, state_vect, exploration_factor, max_iter):
        for i in range(max_iter):
            print("===================================================================================================")
            train_state, state_vect = self.fit(state_vect, exploration_factor)
            print("OUTPUT STATE", state_vect[0][3:])
            # input()
            if train_state == 0:
                print("Reached in", i, "iterations.")
                time.sleep(30)
                return 0                                        # successful reach
        print("Reach unsuccessful.")
        return -1                                               # unsuccessful reach

    def get_reward(self):
        rd = 1 - 2 * (self.arm.get_distance() / self.arm.max_distance)
        return rd * np.abs(rd)

    @staticmethod
    def _choose_action(action, explore):
        e = [np.random.normal() * explore for i in range(len(action))]
        a = action + e
        a[a > 1] = 1
        a[a < -1] = -1
        return a

    @staticmethod
    def _create_actor(input_dim, output_dim, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_dim=input_dim, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_dim, activation='linear'))

        adam = keras.optimizers.Adam(lr=learning_rate)
        sgd = keras.optimizers.sgd(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    @staticmethod
    def _create_critic(input_dim, output_dim, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_dim=input_dim, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_dim, activation="linear"))

        adam = keras.optimizers.Adam(lr=learning_rate)
        sgd = keras.optimizers.sgd(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model


if __name__ == '__main__':
    cacla = Cacla(None, 9, 6, 5, 5, 0.1, 0.5, 0.5)
    cacla.fit(np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]]), 1.0)
    exit()
