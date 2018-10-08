import vrep
import sys
import numpy as np
import math


class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        # degrees of freedom = number of joints.
        self.joints = self._get_joints(6)
        self.obj = self._get_handle('Sphere')
        self.tip = self._get_handle('redundantRob_manipSphere')

    def get_state(self):
        joint_positions = self._get_joints_positions()
        state = joint_positions
        state = np.reshape(state, (1, state.shape[0]))
        return state

    def _get_joints_positions(self):
        positions = np.array([])
        for j in range(len(self.joints)):
            x = vrep.simxGetJointPosition(self.clientID, self.joints[j],
                                          vrep.simx_opmode_oneshot_wait)
            positions = np.hstack((positions, x[1]))

        return positions

    def _move_joint(self, joint, position):
        vrep.simxSetJointTargetPosition(self.clientID, joint, position, vrep.simx_opmode_oneshot_wait)

    def move_joints(self, action):
        for j in range(len(self.joints)):
            self._move_joint(self.joints[j], action[j])

    def _check_error(self, err, msg):
        if err != vrep.simx_return_ok:
            print(err)
            vrep.simxFinish(-1)  # just in case, close all opened connections
            sys.exit(msg)

    def _get_handle(self, name):
        err, handle = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_oneshot_wait)
        self._checkError(err, 'Failed retrieving ' + name + ' handle...')

        return handle

    def _get_joints(self, degrees_of_freedom):
        j1 = self._get_handle('redundantRob_joint1')
        j2 = self._get_handle('redundantRob_joint2')
        j3 = self._get_handle('redundantRob_joint3')
        j4 = self._get_handle('redundantRob_joint4')
        # j5 = self._get_handle('redundantRob_joint5')
        j6 = self._get_handle('redundantRob_joint6')
        j7 = self._get_handle('redundantRob_joint7')
        if degrees_of_freedom == 1:
            return [j2]
        if degrees_of_freedom == 3:
            return [j2, j4, j6]
        if degrees_of_freedom == 6:
            return [j1, j2, j3, j4, j6, j7]

    def get_distance_1(self, a):
        err, pos = vrep.simxGetObjectPosition(self.clientID, self.obj, self.tip,
                                              vrep.simx_opmode_oneshot_wait)
        self._check_error(err, 'Failed in retrieving distance...')
        distance_from_obj = np.linalg.norm(pos)

        # zmenaaaaaaaaaaa
        # print('self.arm.get_state()')
        # print(self.get_state())
        state = (self.get_state() / 1.8237) * 1.45896
        # print('state')
        # print(state)
        a = (a + 0.136) / ((math.pi / 2) + 1.842)
        a = (a - 0.5) * 2.72
        angle_difference = abs(a - state)

        # return distance_from_obj
        return angle_difference
        # zmenaaaaaaaaaaaa

    def get_distance(self):
        err, pos = vrep.simxGetObjectPosition(self.clientID, self.obj, self.tip, vrep.simx_opmode_oneshot_wait)
        self._checkError(err, 'Failed in retrieving distance...')
        distance_from_obj = np.linalg.norm(pos)

        return distance_from_obj

    def get_dist(self):
        err, pos = vrep.simxGetObjectPosition(self.clientID, self.obj, self.tip, vrep.simx_opmode_oneshot_wait)
        self._check_error(err, 'Failed in retrieving distance...')
        # distance_from_obj = np.linalg.norm(pos)
        poss = np.array(pos)
        return poss

    # TODO: check what the two functions below do and when they are called.
    def random_target_position(self):
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

    def set_target_position(self, pos1, pos2, pos3):
        radius = pos1 * 0.6364
        phi = pos2 * 2 * math.pi
        theta = pos3 * math.pi / 2
        x = radius * math.sin(theta) * math.cos(phi) - 0.925
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta) + 0.18944
        vrep.simxSetObjectPosition(self.clientID, self.obj, -1, [x, y, z], vrep.simx_opmode_oneshot_wait)

    def target_position(self):
        distance = np.asarray(vrep.simxGetObjectPosition(self.clientID, self.obj, -1, vrep.simx_opmode_oneshot_wait)[1])
        return distance

    def teaching_signal(self, a):
        error1, desired = vrep.simxGetObjectPosition(self.clientID, self.obj, -1, vrep.simx_opmode_oneshot_wait)
        error2, arm = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_oneshot_wait)

        desired_position = np.array(desired)
        arm_position = np.array(arm)

        difference = desired_position - arm_position
        abs_difference = np.absolute(difference)
        maximum = abs_difference.max()
        # move forward or backward

        x = np.zeros(6)
        for i in range(3):
            if maximum == abs_difference[i]:
                if difference[i] < 0:
                    x[i*2] = a
                elif difference[i] > 0:
                    x[i*2 + 1] = a
                break
        return x

        # The code above should do the same as the one below.
        """if maximum == np.absolute(difference[0]):
            # x = np.array([1,0,0,0,0])
            # return x
            # move backward
            if difference[0] < 0:
                x = np.array([a * 1, 0, 0, 0, 0, 0])
                return x
            # move forward
            if difference[0] > 0:
                x = np.array([0, a * 1, 0, 0, 0, 0])
                return x

        # move right or left
        if maximum == np.absolute(difference[1]):
            # move right
            if difference[1] < 0:
                x = np.array([0, 0, a * 1, 0, 0, 0])
                return x
            # move left
            if difference[1] > 0:
                x = np.array([0, 0, 0, a * 1, 0, 0])
                return x

        if maximum == np.absolute(difference[2]):
            # move down
            if difference[2] < 0:
                x = np.array([0, 0, 0, 0, a * 1, 0])
                return x
            # move up
            if difference[2] > 0:
                x = np.array([0, 0, 0, 0, 0, a * 1])
                return x"""

    def teaching_signal_2(self):
        dist = ArmController.get_dist(self)
        x = np.zeros((6, 1))
        if dist[0] < 0:
            x[0] = dist[0]
        else:
            x[1] = dist[0]

        if dist[1] < 0:
            x[2] = dist[1]
        else:
            x[3] = dist[1]

        if dist[2] < 0:
            x[4] = dist[2]
        else:
            x[5] = dist[2]

        return x

    def joint_modes_for_testing(self):
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd1", "joint1Inv", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd2", "joint2Inv", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd3", "joint3Inv", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd4", "joint4Inv", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd5", "joint5Dpnd", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd6", "joint6Inv", vrep.simx_opmode_oneshot)
        vrep.simxSetStringSignal(self.clientID, "jointModeCmd7", "joint7Inv", vrep.simx_opmode_oneshot)

    def get_reward_quadratic(self):
        err, pos = vrep.simxGetObjectPosition(self.clientID, self.obj, self.tip, vrep.simx_opmode_oneshot_wait)
        self._check_error(err, 'Failed in retrieving distance...')
        r = np.linalg.norm(pos)
        d_max = 1.56
        r_new = 1 - 2 * (r / d_max)
        if r < 0.1:
            return 5
        else:
            return r_new * r_new * np.sign(r_new)




