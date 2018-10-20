import vrep
import random
import sys
import time
import math
import numpy as np

#clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)


class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        self.joints = None  # self._get_joints(6)
        _, self.armHandle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher', vrep.simx_opmode_oneshot_wait)
        _, self.objectHandle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_oneshot_wait)  # self._get_handle('Sphere')
        _, self.tip = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_gripperClose_joint', vrep.simx_opmode_oneshot_wait)  # self._get_handle('redundantRob_manipSphere')

    def reset_arm_position(self):
        vrep.simxSetObjectPosition(self.clientID, self.armHandle, -1, (0, 0, 0.042200), vrep.simx_opmode_streaming)
        self.joints_move([0, 0, 0, 0])

    def reset_object_position(self):
        x = random.randrange(-100, 100) / 1000
        y = random.randrange(200, 300) / 1000
        z = 0.0250
        # x, y, z = 0, 0.25,0.0250 # Default position
        vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_blocking)

    def train(self, model, n_epochs, max_iter, exploration_factor):
        # state_vect = get state vector
        # if it makes more sense for you, you can also move this function to main.py
        model.fit_iter(np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]]), exploration_factor, max_iter)
        pass

    def joints_move(self, joint_angles):
        _, joint1_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint1', vrep.simx_opmode_oneshot_wait)
        _, joint2_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint2', vrep.simx_opmode_oneshot_wait)
        _, joint3_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint3', vrep.simx_opmode_oneshot_wait)
        _, joint4_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint4', vrep.simx_opmode_oneshot_wait)
        handles = [joint1_handle, joint2_handle, joint3_handle, joint4_handle]
        for angle, handle in zip(joint_angles, handles):
            vrep.simxSetJointPosition(clientID, handle, angle, vrep.simx_opmode_oneshot)

    def joints_position(self):
        pass

    def get_distance(self):
        _, [xa, ya, za] = vrep.simxGetObjectPosition(self.clientID, self.objectHandle, -1, vrep.simx_opmode_blocking)
        _, [xb, yb, zb] = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_blocking)
        return math.sqrt(pow((xa - xb), 2) + pow((ya - yb), 2) + pow((za - zb), 2))


if __name__ == "__main__":
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    arm = ArmController(clientID)
    arm.reset_object_position()
    arm.joints_move([0, 0, 0, 0])
    print(arm.get_distance())

    time.sleep(1)
    vrep.simxFinish(clientID)

    exit()
