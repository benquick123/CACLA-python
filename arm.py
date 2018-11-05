import vrep
import random
import sys
import time
import math
import pickle
import numpy as np

#clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)


class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        self.joints = None  # self._get_joints(6)
        _, self.armHandle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher', vrep.simx_opmode_oneshot_wait)
        _, self.objectHandle = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_oneshot_wait)  # self._get_handle('Sphere')
        _, self.tip = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_gripperClose_joint', vrep.simx_opmode_oneshot_wait)  # self._get_handle('redundantRob_manipSphere')
        self.max_distance = 0.9
        self.joint_handles = [
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint1', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint2', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint3', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint4', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint5', vrep.simx_opmode_oneshot_wait)[1]
        ]

    def reset_arm_position(self):
        vrep.simxSetObjectPosition(self.clientID, self.armHandle, -1, (0, 0, 0.042200), vrep.simx_opmode_streaming)
        self.joints_move([0.0] * 5)
        return [0.0] * 5

    def above_floor(self):
        # CAUTION: Only works if bounding box of the object has an absolute reference (Edit > Reorient bbox > world)
        return vrep.simxGetObjectFloatParameter(self.clientID, self.armHandle, vrep.sim_objfloatparam_modelbbox_min_z,
                                                vrep.simx_opmode_blocking)[1] > -0.0423

    def reset_object_position(self):
        # x, y, z = 0, 0.25, 0.0250                   # Default position
        """
        x = random.randrange(-100, 100) / 1000
        y = random.randrange(200, 300) / 1000
        """
        alpha = 2 * math.pi * random.random()
        r = 0.2
        x = r * math.cos(alpha)
        y = r * math.sin(alpha)
        z = 0.0250
        vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_blocking)
        return x, y, z

    def train(self, model, n_epochs, max_iter, exploration_factor):
        for n in range(n_epochs):
            state_vect = [0.0] * 5
            _, object_vect = vrep.simxGetObjectPosition(self.clientID, self.objectHandle, -1, vrep.simx_opmode_blocking)
            # if it makes more sense for you, you can also move this function to main.py
            ef = exploration_factor * (1 - n / n_epochs)
            val = model.fit_iter(np.array([object_vect + state_vect]), ef, max_iter)
            if val == 0:
                pickle.dump(model, open("model_object.pickle", "wb"))
                self.reset_object_position()
            self.reset_arm_position()
            time.sleep(3)
        pass

    def joints_move(self, joint_angles):
        for angle, handle in zip(joint_angles, self.joint_handles):
            vrep.simxSetJointPosition(self.clientID, handle, (angle * math.pi), vrep.simx_opmode_oneshot)

    def joints_move_to_target(self, joint_angles):
        for angle, handle in zip(joint_angles, self.joint_handles):
            vrep.simxSetJointTargetPosition(self.clientID, handle, (angle * math.pi), vrep.simx_opmode_oneshot)

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
    arm.joints_move([0, np.pi/2, 0, 0, 0])
    print(arm.get_distance())

    time.sleep(1)
    vrep.simxFinish(clientID)

    exit()
