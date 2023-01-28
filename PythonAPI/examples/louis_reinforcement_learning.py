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

SHOW_PREVIEW = False 
IM_WIDTH = 640 
IM_HEIGHT = 480 

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMOUNT = 1.0 
    im_width = IM_WIDTH
    im_height = IM_HEIGHT 
    front_camera = None 

    def __init__(self):
        self.client = carla.Client('127.0.0.1',2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world() 
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = [] 
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3,self.transform) 
        self.actor_list.append(self.vehicle) 

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x',f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y',f'{self.im_height}')
        self.rgb_cam.set_attribute('fov',f'110')
        
        transform = carla.Transform(carla.Location(x=2.5,z=0.7)) 
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data:self.process_img(data)) 


def process_img(image):
    i = np.array(image.raw_data) #convert the image into numpy array
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #reshape the image
    i3 = i2[:,:,:3] #Entire height, entire width, rgba values
    cv2.imshow('',i3) #Display the image 
    cv2.waitKey(1) 
    return i3/255.0