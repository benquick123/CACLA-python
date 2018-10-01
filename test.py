import vrep

vrep.simxFinish(-1)
client_id = vrep.simxStart("127.0.0.1", 19997, True, True, 5000, 5)

res, objs = vrep.simxGetObjects(client_id, vrep.sim_handle_all, vrep.simx_opmode_blocking)
print(objs)
print(len(objs))

_, main_camera_handle = vrep.simxGetObjectHandle(client_id, "MainCamera", vrep.simx_opmode_blocking)
_, sphere_handle = vrep.simxGetObjectHandle(client_id, "Sphere", vrep.simx_opmode_blocking)

_, position = vrep.simxGetObjectPosition(client_id, sphere_handle, main_camera_handle, vrep.simx_opmode_blocking)
print(position)

vrep.simxFinish(client_id)
