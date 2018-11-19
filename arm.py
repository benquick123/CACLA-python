import vrep
import random
import sys
import time
import math
import pickle
import numpy as np

# clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)


class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        self.joints = None  # self._get_joints(6)
        _, self.armHandle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher', vrep.simx_opmode_oneshot_wait)
        _, self.objectHandle = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_target', vrep.simx_opmode_oneshot_wait)
        _, self.tip = vrep.simxGetObjectHandle(clientID, 'PhantomXPincher_tip', vrep.simx_opmode_oneshot_wait)
        self.max_distance = 0.9
        self.joint_handles = [
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint1', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint2', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint3', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint4', vrep.simx_opmode_oneshot_wait)[1],
            vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_joint5', vrep.simx_opmode_oneshot_wait)[1]
        ]
        self.object_position_history = []

    def reset_arm_position(self):
        vrep.simxSetObjectPosition(self.clientID, self.armHandle, -1, (0, 0, 0.042200), vrep.simx_opmode_streaming)
        self.joints_move([0.0] * 5)
        return [0.0] * 5

    def above_floor(self):
        # CAUTION: Only works if bounding box of the object has an absolute reference (Edit > Reorient bbox > world)
        return vrep.simxGetObjectFloatParameter(self.clientID, self.armHandle, vrep.sim_objfloatparam_modelbbox_min_z,
                                                vrep.simx_opmode_blocking)[1] > -0.0423

    def no_collision(self):
        # Sometimes works, sometimes doesnt.... something with bounding boxes... I don't know, test before using.

        link1_handle = vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_link1', vrep.simx_opmode_oneshot_wait)[1]
        min_x1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_x, vrep.simx_opmode_blocking)[1]
        max_x1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_x, vrep.simx_opmode_blocking)[1]
        min_y1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_y, vrep.simx_opmode_blocking)[1]
        max_y1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_y, vrep.simx_opmode_blocking)[1]
        min_z1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1]
        max_z1 = vrep.simxGetObjectFloatParameter(self.clientID, link1_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_z, vrep.simx_opmode_blocking)[1]
        x1, y1, z1 = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_blocking)[1]

        link2_handle = vrep.simxGetObjectHandle(self.clientID, 'PhantomXPincher_link2', vrep.simx_opmode_oneshot_wait)[1]
        min_x2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_x, vrep.simx_opmode_blocking)[1]
        max_x2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_x, vrep.simx_opmode_blocking)[1]
        min_y2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_y, vrep.simx_opmode_blocking)[1]
        max_y2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_y, vrep.simx_opmode_blocking)[1]
        min_z2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_min_z, vrep.simx_opmode_blocking)[1]
        max_z2 = vrep.simxGetObjectFloatParameter(self.clientID, link2_handle,
                                                  vrep.sim_objfloatparam_modelbbox_max_z, vrep.simx_opmode_blocking)[1]
        x2, y2, z2 = vrep.simxGetObjectPosition(self.clientID, self.tip, -1, vrep.simx_opmode_blocking)[1]


        return False if min_x1 < x1 < max_x1 and min_y1 < y1 < max_y1 and min_z1 < z1 < max_z1 \
            or min_x2 < x2 < max_x2 and min_y2 < y2 < max_y2 and min_z2 < z2 < max_z2 else True


    def reset_object_position(self):
        # x, y, z = 0, 0.25, 0.0250                   # Default position
        """
        x = random.randrange(-100, 100) / 1000
        y = random.randrange(200, 300) / 1000
        """
        alpha = 2 * math.pi * random.random()
        r = 0.2
        x = abs(r * math.cos(alpha))
        y = abs(r * math.sin(alpha))
        z = 0.0250
        vrep.simxSetObjectPosition(self.clientID, self.objectHandle, -1, (x, y, z), vrep.simx_opmode_blocking)
        self.object_position_history.append((x, y, z))
        return x, y, z

    def train(self, model, n_epochs, max_iter, learning_decay, exploration_factor, log=None):
        for n in range(n_epochs):

            if log is not None:
                log.log("*" * 45 + " EPOCH: " + str(n) + ' ' + "*" * 45)

            state_vect = [0.0] * 5
            _, object_vect = vrep.simxGetObjectPosition(self.clientID, self.objectHandle, -1, vrep.simx_opmode_blocking)
            ef = exploration_factor * (1 - n / n_epochs)
            val = model.fit_iter(np.array([object_vect + state_vect]), ef, max_iter, learning_decay, log)

            if val == 0:
                pickle.dump(model, open("model_object.pickle", "wb"))
            self.reset_object_position()
            self.reset_arm_position()

            if log is not None:
                log.write()
            # time.sleep(1)

    def joints_move(self, joint_angles):
        for angle, handle in zip(joint_angles, self.joint_handles):
            vrep.simxSetJointPosition(self.clientID, handle, (angle * math.pi), vrep.simx_opmode_oneshot)

    def joints_move_to_target(self, joint_angles):
        # For in-simulator movements
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
