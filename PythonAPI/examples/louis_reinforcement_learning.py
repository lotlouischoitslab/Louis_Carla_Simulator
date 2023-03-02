import glob
import os
import sys
import numpy as np 
import cv2 
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam 
from keras.models import Model 

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

SHOW_PREVIEW = False #Display camera in CARLA
IM_WIDTH = 640   #Image Width
IM_HEIGHT = 480  #Image Height

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMT = 1.0
    im_width = IM_WIDTH 
    im_height = IM_HEIGHT 

    def __init__(self):
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout()
        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library() 
        self.model_3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist= [] 
        self.actor_list = [] 

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3,self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb.set_attribute('image_size_x',f'{self.im_width}')
        self.rgb.set_attribute('image_size_y',f'{self.im_height}')
        self.rgb.set_attribute('fov',f'110')

        transform = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0)) #apply control
        time.sleep(4) 

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        return self.front_camera 

    def collision_data(self,event):
        self.collision_hist.append(event) 
    

    def process_img(self,image):
        i = np.array(image.raw_data) #convert the image into numpy array
        i2 = i.reshape((self.im_height,self.im_width,4)) #reshape the image
        i3 = i2[:,:,:3] #Entire height, entire width, rgba values
        if self.SHOW_CAM:
            cv2.imshow('',i3) #Display the image 
            cv2.waitKey(1) 
        self.front_camera =  i3/255.0 
    
    def step(self,action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-1*self.STEER_AMT)) 
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0))  
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=self.STEER_AMT)) 

        v = self.vehicle.get_velocity() 
        kmh = int(3.6*np.sqrt(v.x**2 + v.y**2 + v.z**2)) #kilometers per hour 

        if len(self.collision_hist) != 0:
            done = True 
            reward = -200 
        
        elif kmh < 50:
            done = False 
            reward = -1 
        
        else:
            done = False 
            reward = 1 
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True 
        
        return self.front_camera, reward, done, None 
