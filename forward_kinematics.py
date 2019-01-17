from skinematics import rotmat
import timeit
import numpy as np

# To get DH parameters from V-Rep, import models/tools/DH_extractor, and select joint1
#Between 'AL5D_joint1' and 'AL5D_joint2':
d1=0.0710
theta1=-90.0
a1=0.0120
alpha1=-90.0

#Between 'AL5D_joint2' and 'AL5D_joint3':
d2 =0.0000
theta2 =-90.0
a2 =0.1464
alpha2 =0.0

#Between 'AL5D_joint3' and 'AL5D_joint4':
d3 =0.0000
theta3 =-2.0
a3 =0.1809
alpha3 =0.0

#Between 'AL5D_joint4' and 'AL5D_joint5':
d4 =-0.0047
theta4 =-90.0
a4 =0.0004
alpha4 =-90.0

#Between 'AL5D_joint5' and 'AL5D_tip':
d5=0.0879
theta5=-0.0
a5=0.0003
alpha5=-90.0

tm1 = rotmat.dh(theta1, d1, a1, alpha1)
tm2 = rotmat.dh(theta2, d2, a2, alpha2)
tm3 = rotmat.dh(theta3, d3, a3, alpha3)
tm4 = rotmat.dh(theta4, d4, a4, alpha4)
tm5 = rotmat.dh(theta5, d5, a5, alpha5)

#print(np.dot(np.dot(np.dot(np.dot(tm1, tm2), tm3), tm4), tm5))
#print((np.dot(np.dot(tm1, tm2), tm3)))
#print(np.dot(tm2,tm3))
#print(np.dot(tm4, tm3))
#print(np.dot(np.dot(np.dot(tm1, tm2), tm3), tm4))
#print('\n', tm1)

multidot = np.linalg.multi_dot([tm1, tm2, tm3, tm4, tm5])
print(multidot)