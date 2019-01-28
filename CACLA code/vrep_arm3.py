import gym
import os
import vrep
import numpy as np
import pandas as pd
import gym.spaces
from skinematics import rotmat
import time


class VrepArm(gym.Env):

    def __init__(self, action_multiplier=1.0, simulation=False):
        """
        initialize variables.
        """
        self.simulation = simulation
        self.clientID = -1

        # get necessary handles and init V-rep if simulation == True
        if self.simulation:
            self.clientID = self.open_connect()
            _, self.armHandle = vrep.simxGetObjectHandle(self.clientID, 'AL5D_base', vrep.simx_opmode_blocking)
            _, self.objectHandle = vrep.simxGetObjectHandle(self.clientID, 'AL5D_target', vrep.simx_opmode_blocking)
            _, self.tip = vrep.simxGetObjectHandle(self.clientID, 'AL5D_tip', vrep.simx_opmode_blocking)
            _, self.lfinger = vrep.simxGetObjectHandle(self.clientID, 'AL5D_joint_finger_l', vrep.simx_opmode_blocking)
            _, self.rfinger = vrep.simxGetObjectHandle(self.clientID, 'AL5D_joint_finger_r', vrep.simx_opmode_blocking)
            joint_names = ["AL5D_joint1", "AL5D_joint2", "AL5D_joint3", "AL5D_joint4", "AL5D_joint5"]
            self.joint_handles = []
            for joint_name in joint_names:
                _, joint_handle = vrep.simxGetObjectHandle(self.clientID, joint_name, vrep.simx_opmode_blocking)
                self.joint_handles.append(joint_handle)

        # initialize DH parameters
        params = pd.read_csv("DH_params.csv", delimiter=" = ", header=None, comment="#", engine="python")
        self.init_params = params.values[:, 1].reshape((-1, 4))[:, [1, 0, 2, 3]]
        self.params = None

        # initialize reward range and joint restrictions
        self.reward_range = (-1.0, 1.0)
        self.joint_restrictions = np.array([[-179.0, 179.0], [-90.0, 90.0], [0.0, 160.0], [-90.0, 90.0], [-179.0, 179.0]])
        self.action_multiplier = action_multiplier

        # ... and initialize action and observation space as per OpenAI Gym environment.
        low = np.array([-1.0] * len(self.joint_restrictions))
        high = np.array([1.0] * len(self.joint_restrictions))
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        low = np.hstack((np.array([-np.inf] * 3), low))
        high = np.hstack((np.array([np.inf] * 3), high))
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.object_position = None
        self.max_distance = 0.9
        self.iteration_n = 0

    def open_connect(self):
        """
        opens scene in v-rep if it's not opened yet.
        returns clientID for further communication.
        """
        if self.simulation:
        # open arm model in V-rep if it is not already open. Works only on Windows and same file structure.
        r = {p_name.split(" ")[0] for p_name in os.popen('tasklist /v').read().strip().split('\n')}
        if "vrep.exe" not in r:
            path_to_vrep = "C:/Program Files/V-REP3/"
            vrep_launcher = "v_repLauncher.exe"
            path_to_scene = "/".join(__file__.split("\\")[:-2]) + "/Scenes"
            scene_name = "al5d_scene_forward_kinematics.ttt"

            curr_dir = os.getcwd()

            os.chdir(path_to_vrep)
            os.popen(" ".join([vrep_launcher, "/".join([path_to_scene, scene_name])]))
            os.chdir(curr_dir)

        # connect to v-rep
        clientID = -1
        while clientID == -1:
            vrep.simxFinish(-1)
            clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 1)

        return clientID
        else:
            return -1

    def get_tip_position(self):
        """
        uses parameters to construct DH matrices and multiplies them.
        returns tip_position relative to origin of the base.
        """
        tms = []
        for i in range(len(self.joint_restrictions)):
            tms.append(rotmat.dh(*self.params[i]))

        tm_multidot = np.linalg.multi_dot(tms)
        tip_position = tm_multidot[:3, -1]

        return tip_position

    def reset(self):
        """
        resets arm at the beginning of the episode.
        also sets new target position, and constructs new observation.
        """
        # resets arm position
        self.params = np.array(self.init_params, copy=True)
        self.set_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0])

        # resets object
        alpha = 2 * np.pi * np.random.random()
        r = np.random.uniform(0.10, 0.20)
        x = r * np.cos(alpha)
        y = abs(r * np.sin(alpha))
        z = np.random.uniform(0.0125, 0.1)
        if self.simulation:
            vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_oneshot)
        self.object_position = np.array([x, y, z])
        self.iteration_n = 0

        observation = np.hstack((self.object_position, self.get_joint_positions()))
        return observation

    def render(self, mode='human'):
        """
        renders arm movements in V-rep.
        call only in case self.simulation == True
        """
        if self.params is None:
            print("Call reset() first!")
            exit()

        for joint_params, joint_init_params, joint_handle in zip(self.params, self.init_params, self.joint_handles):
            joint_position = ((joint_params[0] - joint_init_params[0]) * np.pi) / 180.0
            vrep.simxSetJointPosition(self.clientID, joint_handle, joint_position, vrep.simx_opmode_oneshot)

    def step(self, action, absolute=False, move=False):
        """
        makes a step.
        returns new observation, reward, whether episode is done or not and (optional) additional info.
        for more information see OpenAI Gym documentation.
        """
        # limit the joint actions to interval [-1, 1]
        action = np.array(action)
        action[action > 1.0] = 1.0
        action[action < -1.0] = -1.0

        if absolute:
            # when using absolute actions, action equals to new joint positions.
            joint_positions1 = action
            actual_action = [action]
        else:
            # when using relative actions, joint_positions1 = joint_positions0 + (action * action_multiplier)
            joint_positions0 = self.get_joint_positions()
            joint_positions1 = joint_positions0 + (action * np.array([self.action_multiplier]))
            # limit the new joint positions to interval [-1, 1]
            joint_positions1[joint_positions1 > 1.0] = 1.0
            joint_positions1[joint_positions1 < -1.0] = -1.0
            # and save actual_action in case it is different from originally explored one.
            # scaling of action back to [-1, 1] is necessary for training.
            actual_action = [joint_positions1 - joint_positions0] * (np.array([1.0 / self.action_multiplier]))

        self.set_joint_positions(joint_positions1)
        self.iteration_n += 1

        # calculate return values
        observation = np.hstack((self.object_position, self.get_joint_positions()))

        distance = self.get_distance()
        rd = 1 - 2 * (distance / self.max_distance)
        reward = rd * np.abs(rd)

        done = True if distance < 0.01 or self.iteration_n > 100 else False

        # append distance to target in new state and actual action for further processing.
        info = {"distance": distance, "actual_action": actual_action}

        return observation, reward, done, info

    def get_joint_positions(self):
        """
        returns joint positions, scaled to interval [-1, 1]
        """
        joint_positions = []
        for i, (joint_params, joint_restriction) in enumerate(zip(self.params, self.joint_restrictions)):
            joint_position = joint_params[0] - self.init_params[i][0]
            old_joint_range = joint_restriction[1] - joint_restriction[0]
            new_joint_range = self.action_space.high[0] - self.action_space.low[0]
            joint_position = (((joint_position - joint_restriction[0]) * new_joint_range) / old_joint_range) + self.action_space.low[0]
            joint_positions.append(joint_position)

        return np.array(joint_positions)

    def set_joint_positions(self, joint_positions):
        """
        accepts new joint positions and applies changes DH parameters accordingly.
        """
        for i, (joint_position, joint_restriction) in enumerate(zip(joint_positions, self.joint_restrictions)):
            old_joint_range = self.action_space.high[0] - self.action_space.low[0]
            new_joint_range = joint_restriction[1] - joint_restriction[0]
            joint_position = (((joint_position - self.action_space.low[0]) * new_joint_range) / old_joint_range) + joint_restriction[0]
            self.params[i][0] = self.init_params[i][0] + joint_position

    def get_distance(self):
        """
        calculates Euclidean distance between object and tip position.
        """
        if self.object_position is None:
            self.reset()

        object_position = self.object_position
        tip_position = self.get_tip_position()

        return np.sqrt(np.sum(np.power(object_position - tip_position, 2)))

    def close(self):
        """
        closes OpenAI Gym environment and vrep.exe process.
        """
        gym.Env.close(self)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        os.system("TASKKILL /F /IM vrep.exe")


if __name__ == "__main__":
    # some testing code. irrelevant for training itself.
    arm = VrepArm(simulation=True)
    print("RESET", arm.reset()[3:], sep="\n")
    """arm.render()
    print("vrep")
    for i in range(5):
        _, _joint_position = vrep.simxGetJointPosition(arm.clientID, arm.joint_handles[i], vrep.simx_opmode_blocking)
        print((_joint_position * 180.0) / np.pi, end=", ")
    print()
    print(arm.get_joint_positions())
    print(arm.params.shape)
    print(arm.init_params[:, 0])

    print("tip", arm.get_tip_position())
    _, tip_position = vrep.simxGetObjectPosition(arm.clientID, arm.tip, -1, vrep.simx_opmode_blocking)
    print("vrep_tip", tip_position)

    exit()
    input("ALL 1")
    arm.set_joint_positions([0.0] * 3 + [-1.0] + [0.0])
    print(arm.get_joint_positions())
    arm.render()
    print("vrep")
    for i in range(5):
        _, _joint_position = vrep.simxGetJointPosition(arm.clientID, arm.joint_handles[i], vrep.simx_opmode_blocking)
        print((_joint_position * 180.0) / np.pi, end=", ")
    print()
    print("tip", arm.get_tip_position())
    _, tip_position = vrep.simxGetObjectPosition(arm.clientID, arm.tip, -1, vrep.simx_opmode_blocking)
    print("vrep_tip", tip_position)

    exit()"""

    dist = []
    for i in range(1000):
        joint_positions = (np.random.rand(5) * 2.0) - 1.0
        print(joint_positions)
        arm.set_joint_positions(joint_positions)
        print(arm.get_joint_positions())
        print()
        arm.render()

        DH_tip = arm.get_tip_position()
        _, tip = vrep.simxGetObjectHandle(arm.clientID, 'AL5D_tip', vrep.simx_opmode_blocking)
        _, real_tip = vrep.simxGetObjectPosition(arm.clientID, tip, -1, vrep.simx_opmode_blocking)
        real_tip = np.array(real_tip)
        d = np.sqrt(np.sum(np.power(DH_tip - real_tip, 2)))
        dist.append(d)
        print(DH_tip)
        print(real_tip)
        time.sleep(3)

    print("mean:", np.mean(dist))
    print("min:", np.min(dist))
    print("max:", np.max(dist))
    print("std:", np.std(dist))
    exit()

    exit()
