import glob
import os
import sys
import numpy as np 
import cv2 

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time

IM_WIDTH = 640
IM_HEIGHT = 480

actor_list = []

def process_img(image):
    i = np.array(image.raw_data) #convert the image into numpy array
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #reshape the image
    i3 = i2[:,:,:3] #Entire height, entire width, rgba values
    cv2.imshow('',i3) #Display the image 
    cv2.waitKey(1) 
    return i3/255.0

try:
    louis_computer_host = '127.0.0.1'
    client = carla.Client(louis_computer_host,2000)
    client.set_timeout(2.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0] #filter the model by model3
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points()) #random spawn points

    vehicle = world.spawn_actor(bp, spawn_point)
    #vehicle.set_autopilot(True) 

    vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0)) #make car go straight
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x',f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y',f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov','110')

    spawn_point = carla.Transform(carla.Location(x=2.5,z=0.7))

    sensor = world.spawn_actor(cam_bp,spawn_point,attach_to=vehicle) #call the sensor
    actor_list.append(sensor) #add the sensor to the actor list
    sensor.listen(lambda data: process_img(data)) #call the image sensing function

    time.sleep(5)



finally:
    for actor in actor_list:
        actor.destroy()
    print('All cleaned up!')


