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

    def fit(self, state_vect_t0, exploration_factor, log=None):
        # for now exploration will be linear function
        A_t0 = self.actor.predict(state_vect_t0, batch_size=1)
        A_t0 = np.array(A_t0).flatten()
        _exploration_probability = self.exploration_probability * exploration_factor * (self.arm.get_distance() / self.arm.max_distance)
        print(_exploration_probability)
        

        a = self._choose_action(A_t0, _exploration_probability)
        state_vect_t1 = np.reshape(np.append(state_vect_t0[0][:3], a), (1, -1))
        # state_vect_t1[0][3:][state_vect_t1[0][3:] > 1] = 1
        # state_vect_t1[0][3:][state_vect_t1[0][3:] < -1] = -1

        if log is not None:
            log.print("CURRENT STATE: " + str(state_vect_t0[0][3:]))
            log.print("default action: " + str(A_t0))
            log.print("EXPLORING ACTION: " + str(a))

        self.arm.joints_move(a)
        r_t1 = self.get_reward()
        self.arm.joints_move(state_vect_t0[0][3:])

        V_t0 = self.critic.predict(state_vect_t0, batch_size=1)[0][0]
        V_t1 = self.critic.predict(state_vect_t1, batch_size=1)[0][0]

        delta = r_t1 + self.gamma * V_t1 - V_t0

        if log is not None:
            log.print("REWARD: " + str(r_t1))
            log.print("DELTA: " + str(delta))

        self.critic.fit(state_vect_t0, [[r_t1 + self.gamma * V_t1]], verbose=0)

        state_vect = state_vect_t0
        r = None

        if delta > 0:
            self.actor.fit(state_vect_t0, np.array([a.tolist()]), verbose=0)
            # print(state_vect_t1)
            # A_t1 = self.actor.predict(state_vect_t0, batch_size=1, verbose=0)
            # state_vect_t1 = np.array([np.append(state_vect_t0[0][:3], A_t1[0])])
            # print(state_vect_t1)
            if log is not None:
                log.print("-----------------------------------------UPDATING ACTOR--------------------------------------------")
                log.print("PERFORMED ACTION: " + str(a))
            self.arm.joints_move(a)
            state_vect = state_vect_t1
            r = self.get_reward()

        """else:
            if log is not None:
                log.print("-----------------------------------------UPDATING ACTOR--------------------------------------------")
                log.print("PERFORMED ACTION: " + str(A_t0))
            self.arm.joints_move(A_t0)
            state_vect = np.array([np.append(state_vect_t0[0][:3], A_t0)])"""

        if self.arm.get_distance() < 0.02:                      # if distance is smaller than 2 cm
            return 0, r, state_vect                             # 0 = "done"
        else:
            return 1, r, state_vect                             # 1 = "in progress"

    def predict(self, state_vect_t0, move=True):
        A = self.actor.predict(state_vect_t0)
        A[0][A[0] > 1] = 1
        A[0][A[0] < -1] = -1
        if move:
            self.arm.joints_move(A[0])
            r = self.arm.get_distance()
            return r, A
        return A

    def fit_iter(self, state_vect, exploration_factor, max_iter, learning_decay, log=None):
        reward = []
        for i in range(max_iter):
            if log is not None:
                log.print("===================================================================================================")
            train_state, r, state_vect = self.fit(state_vect, exploration_factor, log)
            if r is not None:
                reward.append(r)
            if train_state == 0:
                if log is not None:
                    log.log("Reached in " + str(i) + " iterations.")
                    log.log("AVERAGE REWARD: " + str(np.mean(reward)) + ".")
                time.sleep(1)
                return 0                                        # successful reach

        if log is not None:
            log.log("Reach unsuccessful.")
            log.log("AVERAGE REWARD: " + str(np.mean(reward)) + ".")

        keras.backend.set_value(self.critic.optimizer.lr, keras.backend.get_value(self.critic.optimizer.lr) * learning_decay)
        keras.backend.set_value(self.actor.optimizer.lr, keras.backend.get_value(self.actor.optimizer.lr) * learning_decay)
        # self.critic.optimizer.lr.set_value(self.critic.optimizer.lr.get_value() * learning_decay)
        # self.actor.optimizer.lr.set_value(self.actor.optimizer.lr.get_value() * learning_decay)

        return -1                                               # unsuccessful reach

    def get_reward(self):
        rd = 1 - 2 * (self.arm.get_distance() / self.arm.max_distance)
        return rd * np.abs(rd)
        # return -self.arm.get_distance()

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
        # sgd = keras.optimizers.sgd(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    @staticmethod
    def _create_critic(input_dim, output_dim, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_dim=input_dim, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_dim, activation="linear"))

        adam = keras.optimizers.Adam(lr=learning_rate)
        # sgd = keras.optimizers.sgd(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model


if __name__ == '__main__':
    cacla = Cacla(None, 9, 6, 5, 5, 0.1, 0.5, 0.5)
    cacla.fit(np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]]), 1.0)
    exit()
