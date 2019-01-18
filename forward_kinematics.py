from skinematics import rotmat
import numpy as np

# To get DH parameters from V-Rep, import models/tools/DH_extractor, and select joint1
# Between 'AL5D_joint1' and 'AL5D_joint2':
d1 = 0.0710
theta1 = 90.0
a1 = 0.0120
alpha1 = -90.0

# Between 'AL5D_joint2' and 'AL5D_joint3':
d2 = 0.0000
theta2 = -90.0
a2 = 0.1464
alpha2 = 0.0

# Between 'AL5D_joint3' and 'AL5D_joint4':
d3 = 0.0000
theta3 = 0.0
a3 = 0.1809
alpha3 = 0.0

# Between 'AL5D_joint4' and 'AL5D_joint5':
d4 = -0.0036
theta4 = -90.0
a4 = 0.0004
alpha4 = -90.0

# Between 'AL5D_joint5' and 'AL5D_joint_finger_l':
d5 = 0.0834
theta5 = -0.0
a5 = 0.0000
alpha5 = 90.0

# Between 'AL5D_joint5' and 'AL5D_joint_finger_r':
d6 = 0.0835
theta6 = 180.0
a6 = 0.0000
alpha6 = 90.0


tm1 = rotmat.dh(theta1, d1, a1, alpha1)
tm2 = rotmat.dh(theta2, d2, a2, alpha2)
tm3 = rotmat.dh(theta3, d3, a3, alpha3)
tm4 = rotmat.dh(theta4, d4, a4, alpha4)
tm5 = rotmat.dh(theta5, d5, a5, alpha5)
tm6 = rotmat.dh(theta6, d6, a6, alpha6)


# multi_dot == np.dot(np.dot(np.dot(np.dot(tm1, tm2), tm3), tm4), tm5)
multidot = np.linalg.multi_dot([tm1, tm2, tm3, tm4, tm5])
print(multidot)
#multidot = multidot * [0, 0, 0, 1]
#print(multidot)


x = multidot.item((0, 3))
y = multidot.item((1, 3))
z = multidot.item((2, 3))
print(x, y, z)