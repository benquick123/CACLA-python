import numpy as np
import vrep
import keras

from arm import ArmController
from keras.models import Sequential
from keras.layers import Dense
import time
import math
import os


class Cacla:
    def __init__(self, clientID, number_of_neurons_in_critic, number_of_neurons_in_actor, learning_rate_actor, learning_rate_critic, type_of_exploration, initial_exploration_probability,
                 type_of_reward, type_of_output, discount_factor, exploration_factor, number_of_epochs, type_of_cues, successful_reaches_until_target_change, type_of_action, scale):

        self.clientID = clientID
        self.number_of_neurons_in_critic = number_of_neurons_in_critic
        self.number_of_neurons_in_actor = number_of_neurons_in_actor
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.type_of_exploration = type_of_exploration
        self.initial_exploration_probability = initial_exploration_probability
        self.type_of_reward = type_of_reward
        self.type_of_output = type_of_output
        self.discount_factor = discount_factor
        self.exploration_factor = exploration_factor
        self.number_of_epochs = number_of_epochs
        self.type_of_cues = type_of_cues
        self.succesful_reaches_until_target_change = successful_reaches_until_target_change
        self.type_of_action = type_of_action
        self.scale = scale

        self.arm = ArmController(clientID)
        self.obj = self.arm.obj
        self.state = self.arm.get_state()

        self.output_dimension = 6
        if self.type_of_cues == 'linguistic':
            self.input_dimension = 12
        elif self.type_of_cues == 'continuous':
            self.input_dimension = 9

        # Action approximating NN - approximates policy
        self.actor = self._create_actor(self.input_dimension, self.output_dimension, self.number_of_neurons_in_actor, self.learning_rate_actor)
        # Critic approximating NN - approximates V(s)
        self.critic = self._create_critic(self.input_dimension, self.number_of_neurons_in_critic, self.learning_rate_critic)

    def train(self):
        # change for learning or testing
        load_from = r'\\?\C:\Program Files\V-REP3\V-REP_PRO_EDU\programming\remoteApiBindings\python\python\_1D_1DoF_5NiC_6NiA_0.05lrA_0.09lrC_sigmoidAF_newexplo_SGDOA_quadraticToR_stateToO_0.95DF_0.8EF_10000NoE_continuousToC_1SRuCH_0.9IExP_changeToA_1Scl_23.24_02.05.2017'
        # self.actor.load_weights(load_from + r'\_Actor10000.h5')
        # self.critic.load_weights(load_from + r'\_Critic10000.h5')
        ep = 0
        ep_2 = 0
        succesful_reach = 0
        steps_to_reach = 0
        steps_to_reach_all = np.array([])
        # actions = np.array([])
        actions = np.zeros(self.output_dimension)
        actions_after_fit = np.zeros(self.output_dimension)
        actions_explored = np.zeros(self.output_dimension)
        states = np.zeros(self.output_dimension)
        utility_values_1 = np.array([0])
        utility_values_2 = np.array([0])
        utility_values_after_fit = np.array([0])
        delta_values = np.array([0])
        reward = np.array([0])
        # distance_from_target = np.array([self.arm.get_distance()])
        distance_from_target = np.array([0])  # zmenaaaaaaaaaaa
        exploration_probability = self.initial_exploration_probability * 0  # change in testing/learning
        position_of_target = self.arm.target_position_3D()
        # initial_state = self.state

        if self.type_of_action == 'state':
            def _perform_action(self, action):
                self.arm.move_joints(action)
                self.state = self.arm.get_state()
        elif self.type_of_action == 'change':
            def _perform_action(self, action):
                new_state = action + self.arm.get_state()
                self.arm.move_joints(new_state[0])
                self.state = self.arm.get_state()
        else:
            print("Incorrect type of action.")
            exit()

        if self.type_of_reward == 'linear':
            def get_reward(self):
                r = self.arm.get_distance()
                if r < 0.1:
                    return 10
                else:
                    return 2 - 5 * r
        elif self.type_of_reward == 'quadratic':
            def get_reward(self):
                r = self.arm.get_distance()
                d_max = 1.56
                r_new = 1 - 2 * (r / d_max)
                if self.arm.get_distance() < 0.1:
                    return 5
                else:
                    return r_new * r_new * np.sign(r_new)
        else:
            print("Incorrect type of reward.")
            exit()

        if self.type_of_cues == 'linguistic':
            def teaching_signal():
                return self.arm.teaching_signal(1)

        if self.type_of_cues == 'continuous':
            def teaching_signal():
                return self.arm.target_position()

        def new_random_state():
            # TODO: adjust these values, also check how it applies to used robotic arm.
            return np.array([np.random.rand() * 2 * math.pi, (np.random.rand() - 0.5) * 2 * ((math.pi / 2) + 0.3),
                             np.random.rand() * 2 * math.pi, (np.random.rand()) * ((math.pi / 2) + 1.4) - 0.7,
                             np.random.rand() * ((math.pi / 2) + 1.4) - 0.7 - math.pi / 2, np.random.rand() * 2 * math.pi])

        def random_target_position(self):
            # TODO: adjust these values
            while True:
                radius = np.random.rand() * 0.6364
                phi = np.random.rand() * 2 * math.pi
                theta = np.random.rand() * math.pi / 2
                if np.random.rand() < radius * radius * math.sin(theta) / 0.405:  # 0.405 = 0.6364*0.6364
                    x = radius * math.sin(theta) * math.cos(phi) - 0.925
                    y = radius * math.sin(theta) * math.sin(phi)
                    z = radius * math.cos(theta) + 0.18944
                    vrep.simxSetObjectPosition(self.clientID, self.obj, -1, [x, y, z], vrep.simx_opmode_oneshot_wait)
                    break
            return np.asarray(vrep.simxGetObjectPosition(self.clientID, self.obj, -1, vrep.simx_opmode_oneshot_wait)[1])

        target_position = random_target_position(self)
        all_target_positions = target_position

        exploration_decrease_factor = math.pow(0.1, 1 / self.number_of_epochs)
        print(math.pow(exploration_decrease_factor, 50000))

        for i in range(self.number_of_epochs):
            # while True:
            # print(teaching_signal())
            ep = ep + 1
            ep_2 = ep_2 + 1
            steps_to_reach = steps_to_reach + 1
            exploration_probability = exploration_probability * exploration_decrease_factor
            # exploration_probability = 0
            # self.critic.save_weights('critic_weights_naive.h5',overwrite = True)

            # teach = self.arm.teaching_signal(1)
            # teach = self.arm.teaching_signal_2()
            # teach = np.reshape(teach, (1, teach.shape[0]))

            a = teaching_signal()
            # a = target_position  # use one or another (first one if you train from scratch)
            a = np.reshape(a, (1, a.shape[0]))
            # print('state')
            # print(self.state)
            signal_for_NNs = np.hstack((self.state, a))

            # Choose action and add exploration noise to it
            explored_A = self._choose_action(signal_for_NNs, exploration_probability, self.scale)
            # print(explored_A)
            # print(explored_A[0:6])
            # print(explored_A[6:12])
            actions = np.vstack((actions, explored_A[self.output_dimension:self.output_dimension * 2]))
            actions_explored = np.vstack((actions_explored, explored_A[0:self.output_dimension]))
            explored_A = explored_A[0:self.output_dimension]
            # print(actions)
            # Observe actual value function in this state

            V_t = self.critic.predict(signal_for_NNs, batch_size=1)
            utility_values_1 = np.hstack((utility_values_1, V_t[0]))

            # print('V_t')
            # print(V_t)
            state = self.state
            # Perform action which moves us to new state

            states = np.vstack((states, self.arm._get_joints_positions()))

            _perform_action(self, explored_A)
            distance_from_target = np.vstack((distance_from_target, self.arm.get_distance()))

            a = teaching_signal()
            a = target_position  # """"dat prec potom"""
            a = np.reshape(a, (1, a.shape[0]))

            signal_for_NNs_2 = np.hstack((self.state, a))  # """chabge a - teaching signal"""
            # Observe value function after performing action
            V_t1 = self.critic.predict(signal_for_NNs_2, batch_size=1)
            utility_values_2 = np.hstack((utility_values_2, V_t1[0]))
            # print('V_t1')
            # print(V_t1)
            # Observe reward
            time.sleep(0.1)
            # print(get_reward(a))
            # print(a)
            r = np.array([get_reward(a)])
            # print('reward:')
            # print(r)
            # print(reward)
            # print(r)
            reward = np.hstack((reward, r))
            # print('r= ' + str(r))
            # print(r + self.discount_factor*V_t1)
            # Update critic
            # print('mark1')
            self.critic.fit(signal_for_NNs, r + self.discount_factor * V_t1, verbose=0, batch_size=1, nb_epoch=1)

            utility_values_after_fit = np.vstack((utility_values_after_fit, self.critic.predict(signal_for_NNs, batch_size=1)))
            # print('signal')
            # print(signal_for_NNs)
            # print('r + self.discount_factor*V_t1')
            # print(r + self.discount_factor*V_t1)
            #  print('mark2')
            # if arm got stuck
            # time.sleep(100)

            # If new action was better
            delta = r + self.discount_factor * V_t1 - V_t
            # print(delta[0].shape)
            # print(delta_values.shape)
            delta_values = np.hstack((delta_values, delta[0]))
            if delta > 0:
                explored_A = np.reshape(explored_A, (1, explored_A.shape[0]))
                # print('explored_A')
                # print(explored_A)
                """
                if len(explored_A[0]) == 6:
                    explored_A[0][0] = (explored_A[0][0] / (2*math.pi))/self.scale
                    explored_A[0][1] = ((explored_A[0][1])/(2*((math.pi/2)+0.3))+0.5)/self.scale
                    explored_A[0][2] = (explored_A[0][2]/(2*math.pi))/self.scale
                    explored_A[0][3] = ((explored_A[0][3]+0.7)/((math.pi/2)+1.4))/self.scale
                    explored_A[0][4] = ((explored_A[0][4]+0.7+math.pi/2)/((math.pi/2)+1.4))/self.scale
                    explored_A[0][5] = (explored_A[0][5]/(2*math.pi))/self.scale
                if len(explored_A[0]) == 3:
                    explored_A[0][0] = ((explored_A[0][0])/(2*((math.pi/2)+0.3))+0.5)/self.scale
                    explored_A[0][1] = ((explored_A[0][1]+0.7)/((math.pi/2)+1.4))/self.scale
                    explored_A[0][2] = ((explored_A[0][2]+0.7+math.pi/2)/((math.pi/2)+1.4))/self.scale
                if len(explored_A[0]) == 1:
                    explored_A[0][0] = ((explored_A[0][0])/(2*((math.pi/2)+0.3))+0.5)/self.scale
                """
                # explored_A = explored_A/self.scale #testtesttesttesttesttesttesttesttesttesttesttesttesttest

                # print('explored_A new')
                # print(explored_A)
                # explored_A = explored_A*0
                # print(explored_A)
                # print('Actor is learning')
                # print(self.actor.predict(signal_for_NNs, batch_size=1))
                # for i in range(0,1000):
                explored_A[0][0] = (explored_A[0][0] + 1) / 2
                self.actor.fit(signal_for_NNs, explored_A, verbose=0, batch_size=1, nb_epoch=1)
                actions_after_fit = np.vstack((actions_after_fit, self.actor.predict(signal_for_NNs, batch_size=1)))
                #    print(i/1000-0.5)
                #    print(self.actor.predict(np.array([[0,i/1000-0.5]]), batch_size=1))
                # time.sleep(15)
            # if r > 5: #zadefinovat podla dotiku
            # print('r2')
            # print(self.arm.get_distance(a))
            if self.arm.get_distance() < 0.1:
                # new_state = np.array([(np.random.rand()-0.5)*2*((math.pi/2)+0.3),np.random.rand()*(math.pi/2+1.4)-0.7,
                #                     np.random.rand()*(math.pi/2+1.4)-(math.pi/2)-0.7])
                new_state = new_random_state(self)
                _perform_action(self, new_state)
                # time.sleep(1)
                succesful_reach = succesful_reach + 1
                steps_to_reach_all = np.hstack((steps_to_reach_all, steps_to_reach))
                steps_to_reach = 0

                if succesful_reach == self.succesful_reaches_until_target_change:  # change for learning or testing
                    # a = np.array([(np.random.rand())*((math.pi/2)+0.9)-0.2,np.random.rand()*0.6+0.6])
                    target_position = random_target_position(self)
                    all_target_positions = np.vstack((all_target_positions, target_position))
                    position_of_target = np.vstack((position_of_target, self.arm.target_position_3D()))
                    # self.arm.random_target_position(a[0],a[1])
                    # a = teaching_signal()
                    # print(a)
                    # a = np.reshape(a, (1, a.shape[0]))
                    succesful_reach = 0
                    # time.sleep(0.5) # delays for 2 seconds

            if steps_to_reach == 50:
                new_state = new_random_state(self)  # ???????????????????????????????????
                # overit get_distance #
                steps_to_reach_all = np.hstack((steps_to_reach_all, steps_to_reach))
                steps_to_reach = 0

            if ep_2 == 1:  # change?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # change for learning or testing
                # print(self.newpath)
                # print('saving weights')

                self.actor.save_weights(self.newpath + r'\_Actor' + str(ep) + '.h5', overwrite=True)
                self.critic.save_weights(self.newpath + r'\_Critic' + str(ep) + '.h5', overwrite=True)

                # print('wights saved')
                # self.critic.save_weights('2D_Arm_critic_3_'+str(ep)+'.h5',overwrite = True)
                # print('np_test')

                np.save(self.newpath + r'\_steps_to_reach_all', steps_to_reach_all)
                np.save(self.newpath + r'\_actions', actions)
                np.save(self.newpath + r'\_actions_explored', actions_explored)
                np.save(self.newpath + r'\_utility_values_1', utility_values_1)
                np.save(self.newpath + r'\_utility_values_2', utility_values_2)
                np.save(self.newpath + r'\_delta_values', delta_values)
                np.save(self.newpath + r'\_reward', reward)
                np.save(self.newpath + r'\_states', states)
                np.save(self.newpath + r'\position_of_target', position_of_target)
                np.save(self.newpath + r'\distance_from_target', distance_from_target)
                np.save(self.newpath + r'\actions_after_fit', actions_after_fit)
                np.save(self.newpath + r'\utility_values_after_fit', utility_values_after_fit)
                np.save(self.newpath + r'\all_target_positions', all_target_positions)

                # np_test = np.load(newpath + r'\2D_Arm_steps_to_reach_all_3_.npy')
                # print(np_test)
                ep_2 = 0

    def _choose_action(self, signal_for_NNs, exploration_probability, scale):
        action = self.actor.predict(signal_for_NNs, batch_size=1)
        explore = np.zeros(self.output_dimension)
        action = (action - 0.5) * 2

        # print(action)
        # action = action*scale # testtesttesttesttesttesttesttesttesttest
        # print(action)

        if np.random.rand() < exploration_probability:
            explore = self.exploration_factor * np.random.randn(self.output_dimension)  # zmenit na gaussovsku, ci menit velkost
            # print(action)
            # print(explore)
        explored_action = action + explore
        result = np.hstack((explored_action[0], action[0]))
        # print(result)
        return result

    def _create_critic(self, dimensions, number_of_neurons_in_critic, learning_rate):
        model = Sequential()
        model.add(Dense(number_of_neurons_in_critic, input_dim=dimensions, activation="sigmoid"))
        model.add(Dense(1))

        sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model

    def _create_actor(self, dimensions, output_dimension, number_of_neurons_in_actor, learning_rate):
        model = Sequential()
        model.add(Dense(number_of_neurons_in_actor, input_dim=dimensions, activation="sigmoid"))
        model.add(Dense(output_dimension, activation='sigmoid'))

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

