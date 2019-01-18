import gym
import os
import vrep
import numpy as np
import gym.spaces
from skinematics import rotmat

class VrepArm(gym.Env):

    def __init__(self, action_multiplier=1.0, simulation=False):
        self.simulation = simulation
        self.clientID = self.open_connect()

        # Denavit-Hartenberg parameters
        # Between 'AL5D_joint1' and 'AL5D_joint2':
        self.d1 = 0.0710
        self.theta1 = 90.0
        self.a1 = 0.0120
        self.alpha1 = -90.0

        # Between 'AL5D_joint2' and 'AL5D_joint3':
        self.d2 = 0.0000
        self.theta2 = -90.0
        self.a2 = 0.1464
        self.alpha2 = 0.0

        # Between 'AL5D_joint3' and 'AL5D_joint4':
        self.d3 = 0.0000
        self.theta3 = 0.0
        self.a3 = 0.1809
        self.alpha3 = 0.0

        # Between 'AL5D_joint4' and 'AL5D_joint5':
        self.d4 = -0.0036
        self.theta4 = -90.0
        self.a4 = 0.0004
        self.alpha4 = -90.0

        # Between 'AL5D_joint5' and 'AL5D_joint_finger_l':
        self.d5 = 0.0834
        self.theta5 = -0.0
        self.a5 = 0.0000
        self.alpha5 = 90.0

        # Between 'AL5D_joint5' and 'AL5D_joint_finger_r':
        self.d6 = 0.0835
        self.theta6 = 180.0
        self.a6 = 0.0000
        self.alpha6 = 90.0

        # initialize necessary variables
        self.reward_range = (-1.0, 1.0)
        self.joint_restrictions = np.array([[-179.0, 179.0], [-45.0, 90.0], [0.0, 160.0],
                                            [-90.0, 90.0], [-179.0, 179.0]])
        self.action_multiplier = action_multiplier

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
        self.last_tip_position = None


    def get_tip(self):
        # DH transition matrices
        tm1 = rotmat.dh(self.theta1, self.d1, self.a1, self.alpha1)
        tm2 = rotmat.dh(self.theta2, self.d2, self.a2, self.alpha2)
        tm3 = rotmat.dh(self.theta3, self.d3, self.a3, self.alpha3)
        tm4 = rotmat.dh(self.theta4, self.d4, self.a4, self.alpha4)
        tm5 = rotmat.dh(self.theta5, self.d5, self.a5, self.alpha5)

        tm_multidot = np.linalg.multi_dot([tm1, tm2, tm3, tm4, tm5])
        tip_x = tm_multidot.item((0, 3))
        tip_y = tm_multidot.item((1, 3))
        tip_z = tm_multidot.item((2, 3))
        return tip_x, tip_y, tip_z

    def reset(self):
        # reset arm
        self.theta1 = 90.0
        self.theta2 = -90.0
        self.theta3 = 0.0
        self.theta4 = -90.0
        self.theta5 = -0.0

        # reset object
        alpha = 2 * np.pi * np.random.random()
        r = 0.15        # np.random.uniform(0.10, 0.20)
        x = abs(r * np.cos(alpha))
        y = abs(r * np.sin(alpha))
        z = 0.0125          # np.random.uniform(0.0125, 0.1)
        self.object_position = np.array([x, y, z])
        self.iteration_n = 0

        observation = np.hstack((self.object_position, self.get_joint_positions()))

        # print("Environment reset.")
        return np.array(observation)

    def render(self, mode='human'):
        # possible use in when simulation == True?
        pass

    def step(self, action, absolute=False, move=False):
        # move joints first
        if absolute:
            action = np.array(action)
            action[action > 1.0] = 1.0
            action[action < -1.0] = -1.0
            joint_positions1 = action
            actual_action = [action]
        else:
            joint_positions0 = self.get_joint_positions()
            joint_positions1 = joint_positions0 + (action * np.array([self.action_multiplier]))
            joint_positions1[joint_positions1 > 1.0] = 1.0
            joint_positions1[joint_positions1 < -1.0] = -1.0
            actual_action = [joint_positions1 - joint_positions0]

        self.set_joint_positions(joint_positions1)
        self.iteration_n += 1

        # calculate return values then
        observation = np.hstack((self.object_position, self.get_joint_positions()))

        distance = self.get_distance()
        rd = 1 - 2 * (distance / self.max_distance)
        reward = rd * np.abs(rd)

        done = True if distance < 0.01 or self.iteration_n > 50 else False

        info = {"distance": distance, "actual_action": actual_action}

        return observation, reward, done, info

    def get_joint_positions(self):
        if self.last_joint_positions is not None and \
                not ((self.last_joint_positions >= 1.001).any() or (self.last_joint_positions <= -1.001).any()):
            return self.last_joint_positions

        joint_positions = [self.theta1, self.theta2, self.theta3, self.theta4, self.theta5]

        self.last_joint_positions = np.array(joint_positions)
        return self.last_joint_positions

    def set_joint_positions(self, joint_positions):
        self.last_joint_positions = np.array(joint_positions)
        self.last_tip_position = None

        self.theta1, self.theta2, self.theta3, self.theta4, self.theta5 = joint_positions


    def get_distance(self):
        if self.object_position is not None:
            object_position = self.object_position
        else:
            object_position = self.reset()
            self.object_position = object_position
        if self.last_tip_position is not None:
            tip_position = self.last_tip_position
        else:
            tip_position = self.get_tip()
            tip_position = np.array(tip_position)
            self.last_tip_position = tip_position
        return np.sqrt(np.sum(np.power(object_position - tip_position, 2)))

    def close(self):
        gym.Env.close(self)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        os.system("TASKKILL /F /IM vrep.exe")


if __name__ == "__main__":
    arm = VrepArm()
    obs = arm.reset()
    obv, _, _, _ = arm.step([0.0, 0.1, 0.1, -0.1, 0.1])
    print("obv", obv)
    print("last", arm.get_joint_positions())
    print("dist", arm.get_distance())
    arm.last_joint_positions = None
    arm.last_tip_position = None
    print("when none", arm.get_joint_positions())
    print("dist none", arm.get_distance())
    print()

    obv, _, _, _ = arm.step([0.0, 0.1, 0.1, -0.1, 0.1])
    print("obv", obv)
    print("last", arm.get_joint_positions())
    print("dist", arm.get_distance())
    arm.last_joint_positions = None
    arm.last_tip_position = None
    print("when none", arm.get_joint_positions())
    print("dist none", arm.get_distance())
    print()

    obv, _, _, _ = arm.step(-np.array([0.0, 0.1, 0.1, -0.1, 0.1]))
    print("obv", obv)
    print("last", arm.get_joint_positions())
    print("dist", arm.get_distance())
    arm.last_joint_positions = None
    arm.last_tip_position = None
    print("when none", arm.get_joint_positions())
    print("dist none", arm.get_distance())
    print()
    # print(arm.get_distance())
