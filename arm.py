import vrep
import random
import sys
import time
import math


clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        _, self.armHandle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher', vrep.simx_opmode_oneshot_wait)
        self.joints = None              # self._get_joints(6)
        _, self.objectHandle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_oneshot_wait)  # self._get_handle('Sphere')
        _, self.tip = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_gripperClose_joint', vrep.simx_opmode_oneshot_wait)                 # self._get_handle('redundantRob_manipSphere')

    def reset_arm_position(self):
        vrep.simxSetObjectPosition(self.clientID, self.armHandle, -1, (0, -0.25, 0.042200), vrep.simx_opmode_streaming)

    def reset_object_position(self):
        # x, y, z = ((random.random()*4) - 2)/10, 0, 0.0250
        x, y, z = 0, 0, 0.0250
        vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_blocking)

    def train(self, n_iterations):
        pass

    def joints_move(self, array):
        [j1, j2, j3, j4] = array
        _, joint1_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint1', vrep.simx_opmode_oneshot_wait)
        _, joint2_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint2', vrep.simx_opmode_oneshot_wait)
        _, joint3_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint3', vrep.simx_opmode_oneshot_wait)
        _, joint4_handle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_joint4', vrep.simx_opmode_oneshot_wait)
        handles = [joint1_handle,joint2_handle, joint3_handle, joint4_handle]
        for value, handle in zip(array, handles):
            vrep.simxSetJointPosition(clientID, handle, value, vrep.simx_opmode_oneshot)


    def joints_position(self):
        pass

    def get_distance(self):
        _, [xa, ya, za] = vrep.simxGetObjectPosition(self.clientID, self.armHandle, -1, vrep.simx_opmode_blocking)
        _, [xb, yb, zb] = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_blocking)
        return math.sqrt(pow((xa - xb), 2) + pow((ya - yb), 2) + pow((za - zb), 2))


arm = ArmController(clientID)
arm.reset_object_position()
arm.joints_move([0,1,-1,0])
print(arm.get_distance())

time.sleep(1)
vrep.simxFinish(clientID)
