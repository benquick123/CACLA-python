import gym
import os
import vrep
import numpy as np
import gym.spaces


class VrepArm(gym.Env):

    def __init__(self, simulation=False):
        self.simulation = simulation
        self.clientID = self.open_connect()

        # get necessary handles
        _, self.armHandle = vrep.simxGetObjectHandle(self.clientID, 'AL5D_base', vrep.simx_opmode_blocking)
        _, self.objectHandle = vrep.simxGetObjectHandle(self.clientID, 'AL5D_target', vrep.simx_opmode_blocking)
        _, self.tip = vrep.simxGetObjectHandle(self.clientID, 'AL5D_tip', vrep.simx_opmode_blocking)

        joint_names = ["AL5D_joint1", "AL5D_joint2", "AL5D_joint3",
                       "AL5D_joint4", "AL5D_joint5"]
        self.joint_handles = []
        for joint_name in joint_names:
            _, joint_handle = vrep.simxGetObjectHandle(self.clientID, joint_name, vrep.simx_opmode_blocking)
            self.joint_handles.append(joint_handle)

        # initialize necessary variables
        self.reward_range = (-1.0, 1.0)
        self.joint_restrictions = np.array([[-180.0, 180.0], [-45.0, 90.0], [0.0, 160.0],
                                            [-90.0, 90.0], [-180.0, 180.0]])

        low = np.array([-1.0] * len(self.joint_restrictions))
        high = np.array([1.0] * len(self.joint_restrictions))
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        low = np.hstack((np.array([-np.inf] * 3), low))
        high = np.hstack((np.array([np.inf] * 3), high))
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.object_position = None
        self.max_distance = 0.9
        self.iteration_n = 0
        self.last_joint_positions = None

    def open_connect(self):
        # open scene in v-rep if it's not opened yet
        r = {p_name.split(" ")[0] for p_name in os.popen('tasklist /v').read().strip().split('\n')}
        if "vrep.exe" not in r:
            path_to_vrep = "C:/Program Files/V-REP3/"
            vrep_launcher = "v_repLauncher.exe"
            path_to_scene = "/".join(__file__.split("\\")[:-2]) + "/Scenes"
            scene_name = "main_scene.ttt"

            curr_dir = os.getcwd()

            os.chdir(path_to_vrep)
            os.popen(" ".join([vrep_launcher, "/".join([path_to_scene, scene_name])]))
            os.chdir(curr_dir)

        # connect to v-rep
        clientID = -1
        while clientID == -1:
            vrep.simxFinish(-1)
            clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 1)

        if self.simulation:
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        return clientID

    def reset(self):
        # reset arm
        vrep.simxSetObjectPosition(self.clientID, self.armHandle, -1, (0, 0, 0.02), vrep.simx_opmode_oneshot)
        joint_positions = np.array([0.0] * self.action_space.shape[0])
        self.set_joint_positions(joint_positions)

        # reset object
        alpha = 2 * np.pi * np.random.random()
        r = np.random.uniform(0.10, 0.25)
        x = r * np.cos(alpha)
        y = abs(r * np.sin(alpha))
        z = np.random.uniform(0.0125, 0.1)
        vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_oneshot)
        self.object_position = np.array([x, y, z])
        self.iteration_n = 0

        observation = self.object_position.tolist() + self.get_joint_positions().tolist()

        # print("Environment reset.")
        return np.array(observation)

    def render(self, mode='human'):
        # possible use in when simulation == True?
        pass

    def step(self, action, absolute=False):
        # move joints first
        if absolute:
            joint_positions1 = action
        else:
            joint_positions0 = self.get_joint_positions()
            joint_positions1 = joint_positions0 + action

        self.set_joint_positions(joint_positions1)
        self.iteration_n += 1

        # calculate return values then
        observation = np.array(self.object_position.tolist() + self.get_joint_positions().tolist())

        distance = self.get_distance()
        rd = 1 - 2 * (distance / self.max_distance)
        reward = rd * np.abs(rd)

        done = True if distance < 0.01 or self.iteration_n > 50 else False

        info = {"distance": distance}

        return observation, reward, done, info

    def set_joint_positions(self, joint_positions):
        self.last_joint_positions = np.array(joint_positions)
        for joint_handle, joint_position, joint_restriction in zip(self.joint_handles, joint_positions, self.joint_restrictions):
            joint_position = joint_position * (joint_restriction[1] / 180.0) * np.pi
            if self.simulation:
                vrep.simxSetJointTargetPosition(self.clientID, joint_handle, joint_position, vrep.simx_opmode_oneshot)
            else:
                vrep.simxSetJointPosition(self.clientID, joint_handle, joint_position, vrep.simx_opmode_oneshot)

    def get_joint_positions(self):
        if self.last_joint_positions is not None and \
                not ((self.last_joint_positions >= 1.001).any() or (self.last_joint_positions <= -1.001).any()):
            return self.last_joint_positions

        joint_positions = []
        for joint_handle, joint_restriction in zip(self.joint_handles, self.joint_restrictions):
            _, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_blocking)
            joint_position = (joint_position * 180.0) / (joint_restriction[1] * np.pi)
            joint_positions.append(joint_position)

        self.last_joint_positions = np.array(joint_positions)
        return self.last_joint_positions

    def get_distance(self):
        _, object_position = vrep.simxGetObjectPosition(self.clientID, self.objectHandle, -1, vrep.simx_opmode_blocking)
        _, tip_position = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_blocking)
        object_position = np.array(object_position)
        tip_position = np.array(tip_position)
        return np.sqrt(np.sum(np.power(object_position - tip_position, 2)))

    def close(self):
        gym.Env.close(self)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        os.system("TASKKILL /F /IM vrep.exe")
