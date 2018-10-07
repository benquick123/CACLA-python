import vrep
import sys
import time
import math

pi = math.pi

vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
if clientID != -1:
    print('Connected to remote API server')
else:
    sys.exit("Could not connect")


ec1, motor1_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor1', vrep.simx_opmode_oneshot_wait)
ec2, motor2_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor2', vrep.simx_opmode_oneshot_wait)
ec3, motor3_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor3', vrep.simx_opmode_oneshot_wait)
ec4, motor4_handle = vrep.simxGetObjectHandle(clientID, 'uarm_motor4', vrep.simx_opmode_oneshot_wait)
ec5, auxMotor2_handle = vrep.simxGetObjectHandle(clientID, 'uarm_auxMotor2', vrep.simx_opmode_oneshot_wait)
ec6, gripperMotor2_handle = vrep.simxGetObjectHandle(clientID, 'uarmGripper_motor2Method2', vrep.simx_opmode_oneshot_wait)

vrep.simxSetJointTargetPosition(clientID, motor1_handle, pi, vrep.simx_opmode_oneshot) #rotates motor1, takes radians, default is at pi/2
vrep.simxSetJointTargetPosition(clientID, motor2_handle, 1, vrep.simx_opmode_oneshot)# motor2 and 3 work simultaneously


"""
#If not sure about commands being received, add a 'pioneer p3dx' to the scene and run this.

_, left_motor_handle = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID, left_motor_handle, 0.2, vrep.simx_opmode_streaming)
"""

time.sleep(1)
