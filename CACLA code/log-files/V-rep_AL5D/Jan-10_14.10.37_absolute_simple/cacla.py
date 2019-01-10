from builtins import object

import numpy as np
import keras
import time

from keras.models import Sequential
from keras.layers import Dense


class Cacla:
    def __init__(self, env, input_dim, output_dim, alpha, beta, gamma, lr_decay, exploration_decay, exploration_factor):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.exploration_factor = exploration_factor
        self.lr_decay = lr_decay
        self.exploration_decay = exploration_decay

        self.alpha = alpha
        self.beta = beta

        self.actor = self._create_actor(input_dim, output_dim, alpha)
        self.critic = self._create_critic(input_dim, 1, beta)

    def update_lr(self, lr_decay):
        """

        :param lr_decay: decay for both actor and critic
        changes learning rate for actor and critic based on lr_decay.
        """
        keras.backend.set_value(self.critic.optimizer.lr,
                                keras.backend.get_value(self.critic.optimizer.lr) * lr_decay)
        keras.backend.set_value(self.actor.optimizer.lr,
                                keras.backend.get_value(self.actor.optimizer.lr) * lr_decay)
        self.alpha *= lr_decay
        self.beta *= lr_decay

    def update_exploration(self, exploration_decay=None):
        """
        updates the exploration factor.
        :param exploration_decay: exploration_factor multiplier. if None, default value is used.
        """
        if exploration_decay is None:
            exploration_decay = self.exploration_decay
        self.exploration_factor *= exploration_decay

    @staticmethod
    def sample(action, explore):
        """
        :param action: default action predicted by actor
        :param explore: exploration factor
        :return: explored action, normally distributed around default action.
        """
        e = [np.random.normal() * explore for i in range(len(action))]
        a = action + e
        return a

    @staticmethod
    def _create_actor(input_dim, output_dim, learning_rate):
        """
        Creates actor. Uses 2 layers with number of neurons described in next 3 lines.
        initializes weights to some small value.
        """
        l1_size = 5 * input_dim
        l3_size = 5 * output_dim
        # l2_size = int(np.sqrt(l1_size * l3_size))

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_dim, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / input_dim))))
        # model.add(Dense(l2_size, activation="relu",
        #                 kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l1_size))))
        model.add(Dense(l3_size, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l1_size))))
        model.add(Dense(output_dim, activation="linear",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l3_size))))

        adam = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    @staticmethod
    def _create_critic(input_dim, output_dim, learning_rate):
        """
        See self._create_actor.
        """
        l1_size = 5 * input_dim
        l3_size = 5 * output_dim
        # l2_size = int(np.sqrt(l1_size * l3_size))

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_dim, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / input_dim))))
        # model.add(Dense(l2_size, activation="relu",
        #                 kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l1_size))))
        model.add(Dense(l3_size, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l1_size))))
        model.add(Dense(output_dim, activation='linear',
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(1 / l3_size))))

        adam = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model
