import vrep
import sys
import numpy as np
import math


class ArmController:
    def __init__(self, clientID):
        self.clientID = clientID
        self.joints = None              # self._get_joints(6)
        self.obj = None                 # self._get_handle('Sphere')
        self.tip = None                 # self._get_handle('redundantRob_manipSphere')

    def reset_arm_position(self):
        pass

    def reset_object_position(self):
        pass

    def train(self, n_iterations):
        pass

    def joints_move(self):
        pass

    def joints_position(self):
        pass

    def get_distance(self):
        pass




