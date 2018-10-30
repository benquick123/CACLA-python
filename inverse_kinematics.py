
# !!! Simulation must be running for IK to work

import vrep
import sys
import time
import random

vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
if clientID != -1:
    print('Connected to remote API server')
else:
    sys.exit("Could not connect")


_, robot_handle = vrep.simxGetObjectHandle(clientID, 'redundantRobot', vrep.simx_opmode_oneshot_wait)
_, target_handle = vrep.simxGetObjectHandle(clientID, 'redundantRob_target', vrep.simx_opmode_oneshot_wait)
vrep.simxSetObjectPosition(clientID, robot_handle, -1, (0, 0, 0.0647), vrep.simx_opmode_blocking)


for i in range(100):
    x = random.random() - 0.5
    y = random.random() - 0.5
    z = 0.25 + random.random()/2
    vrep.simxSetObjectPosition(clientID, target_handle, -1, (x, y, z), vrep.simx_opmode_blocking)
    time.sleep(2)


time.sleep(1)
vrep.simxFinish(clientID)