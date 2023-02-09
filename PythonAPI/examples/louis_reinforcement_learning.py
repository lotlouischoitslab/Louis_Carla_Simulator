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


SHOW_PREVIEW = False #Set show preview to False
IM_WIDTH = 640   #Image Width
IM_HEIGHT = 480  #Image Height
SECONDS_PER_EPISODE = 10 #Seconds per episode
REPLAY_MEMORY_SIZE = 5_000 #5000 replay memory size
MIN_REPLAY_MEMORY_SIZE = 1_000 #Minimum replay memory size
PREDICTION_BATCH_SIZE = 1 #Prediction batch size
TRAINING_BATCH_SIZE = MINIBATCH // 4 
UPDATE_TARGET_EVERY = 5 #At the end of how many we update the model
MODEL_NAME = 'Xception' #Which type of model to use

MEMORY_FRACTION = 0.8 #GPU I want to use 
MIN_REWARD = -200 #Minimum reward 

EPISODES = 100 #Number of episodes

DISCOUNT = 0.99 #Discount Rate
epsilon = 1 #Epsilon value
EPSILON_DECAY = 0.95 #How much we are going to decay
MIN_EPSILON = 0.001 

AGGREGATE_STATS_EVERY = 10 



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

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
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
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-1*self.STEER_AMT)) #set to full throttle and go straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0)) #go right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=self.STEER_AMT))  #go left

        v = self.vehicle.get_velocity() #speed
        kmh = int(3.6*np.sqrt(v.x**2 + v.y**2 + v.z**2)) 

        if len(self.collision_hist) != 0:
            done = True 
            reward = -200 #BIG penalty 
        elif kmh < 50:
            done = False 
            reward = -1 #Minor penalty
        else:
            done = False 
            reward = 1 #We will keep appending one to the total reward
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True 
        
        return self.front_camera, reward, done,None 

class DQAgent:
    def __init__(self):
        self.model = self.create_model() #create the deep q-learning agent
        self.target_model=  self.create_model() #set the target model
        self.target_model.set_weights(self.model.get_weights()) #set the weights of the model

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #replay the memory

        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')
        self.target_update_counter = 0
        self.graph = tf.get_default_graph() 

        self.terminate = False #Flags for our thread
        self.last_logged_episode = 0 #Keep track of our tensorboard

        self.training_initialized = False #When we start running simulation

    
    def create_model(self):
        base_model = Xception(weights=None, include_top=False,input_shape=(IM_HEIGHT,IMG_WIDTH,3)) #set the base model

        x = base_model.output #base model output
        x = GlobalAveragePooling2D()(x) #average pooling

        predictions = Dense(3,activation='linear')(x) #3 options: left, straight, right
        model = Model(inputs=base_model.input,outputs=predictions) #call the model
        model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['accuracy']) #measure the accuracy
        return model #return the model

    
    def update_replay_memory(self,transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition) #replay the memory

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE) #batch of replay memory

        current_states = np.array([transition[0] for transition in minibatch])/ 255 #scale the information

        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states,PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/ 255 #scale the information

        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states,PREDICTION_BATCH_SIZE)

        X = []
        y = [] 

        for index,(current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_future_q
            else:
                new_q = reward 
        
            current_qs = current_qs_list[index] #update the current q
            current_qs[action] = new_q #update the actual q-value

            X.append(current_state) #append the current state
            y.append(current_qs) #append the current qs
        
        log_this_step = False 
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True 
            self.last_logged_episode = self.tensorboard.step 

            
        with self.graph.as_default():
            self.model.fit(np.array(X)/255,np.array(y),batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False,callbacks=[self.tensorboard] if log_this_step else None )

        if log_this_step:
            self.target_update_counter += 1 
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0 

        
    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1*state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1,IM_HEIGHT,IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size=(1,3)).astype(np.float32) 

        with self.graph.as_default():
            self.model.fit(X,y,verbose=False,batch_size=1) 
        
        self.training_initialized = True 

        while True:
            if self.terminate:
                return 

            self.train() 
            time.sleep(0.01)  

