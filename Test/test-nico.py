import vrep
import time
import sys
import math


class NicoController:
    def __init__(self, clientID):
        self.clientID = clientID
        _, self.object = vrep.simxGetObjectHandle(self.clientID, 'Cylinder', vrep.simx_opmode_oneshot_wait)
        _, self.target = vrep.simxGetObjectHandle(self.clientID, 'target', vrep.simx_opmode_oneshot_wait)
        self.joint_handles = [
            vrep.simxGetObjectHandle(self.clientID, 'l_indexfingers_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_indexfinger_1st_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_indexfinger_2nd_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_ringfingers_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_ringfinger_1st_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_ringfinger_2nd_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_thumb_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_thumb_1st_x', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'l_thumb_2nd_x', vrep.simx_opmode_oneshot_wait)[1],
        ]

    def grab(self):
        joint_angles = [-1]*9
        for angle, handle in zip(joint_angles, self.joint_handles):
            vrep.simxSetJointTargetPosition(self.clientID, handle, angle, vrep.simx_opmode_oneshot)

    def release(self):
        joint_angles = [0]*9
        for angle, handle in zip(joint_angles, self.joint_handles):
            vrep.simxSetJointTargetPosition(self.clientID, handle, angle, vrep.simx_opmode_oneshot)

    def get_target_object_coordinates(self):
        return tuple(vrep.simxGetObjectPosition(self.clientID, self.object, -1, vrep.simx_opmode_blocking)[1])

    def move_to(self, coordinate_tuple):
        vrep.simxSetObjectPosition(self.clientID, self.target, -1, coordinate_tuple, vrep.simx_opmode_blocking)


vrep.simxFinish(-1)  # just in case, close all opened connections

clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:  # check if client connection successful
    print('Connected to remote API server')
else:
    sys.exit('Could not connect')


nico = NicoController(clientID)

vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
time.sleep(1)
nico.move_to(nico.get_target_object_coordinates())
time.sleep(2)
nico.grab()
time.sleep(2)
nico.move_to((0.3750, -0.1500, 0.6750))
time.sleep(2)
nico.release()
time.sleep(5)
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

time.sleep(1)
vrep.simxFinish(clientID)
