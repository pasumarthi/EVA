# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from PIL import Image as PILImage


# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty,BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
episode_num = 0

# Getting our AI, which we call "policy", and that contains our neural network that represents our Q-function
policy = TD3((1,40,40),1,10)


last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    
    global sand
    global goal_x
    global goal_y
    global first_update
    global swap
    global done
    global image_step 
    global episode_timesteps
    

    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    first_update = False


    episode_timesteps = 0
    total_timesteps = 0
    
    swap = 0
    done=0
    image_step=0

# first destination 
    goal_x = 1320
    goal_y = 625


# Initializing the last distance
    last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0)
    rotation = BoundedNumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity=Vector(6, 0)
         

    def update(self, dt):

        global policy
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global done
        global image_step
        global episode_timesteps
        global episode_num  
        global new_obs_ori
        global new_obs_dis
        

        longueur = self.width
        largeur =  self.height
        if first_update:
            init()

    #getting car image at the current location.

        # image = sand[int(self.car.x)-20:int(self.car.x)+20, int(self.car.y)-20:int(self.car.y)+20]
       
        #---------------------------cropping-------------------------
        crop_size =40
        sand = np.asarray(PILImage.open("./images/MASK1.png").convert('L'))/255
        pad = crop_size*2
        #pad for safety
        crop1 = np.pad(sand, pad_width=pad, mode='constant', constant_values = 1)
        # centerx = car_x + pad
        # centery = car_y + pad
        centerx = self.car.x + pad
        centery = self.car.y + pad

        #first small crop
        startx, starty = int(centerx-(crop_size)), int(centery-(crop_size))
        crop1 = crop1[starty:starty+crop_size*2, startx:startx+crop_size*2]

        #rotate
        crop1 = scipy.ndimage.rotate(crop1, -self.car.angle, mode='constant', cval=1.0, reshape=False, prefilter=False)
        #again final crop
        startx, starty = int(crop1.shape[0]//2-crop_size//2), int(crop1.shape[0]//2-crop_size//2)
        
        #-------------------cropping---------------------------------
     
        image = crop1[starty:starty+crop_size, startx:startx+crop_size].reshape(crop_size, crop_size)

        # calculate the distance from destaination after current move of car
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
    
        # calculating number of steps of image so that it can be used for random movement
        image_step += 1

        curent_state1 = image
        

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        new_obs_ori = Vector(*self.car.velocity).angle((xx,yy))/180.0          
        new_obs_dis = distance
        
        current_state2 = [new_obs_ori,-new_obs_ori,new_obs_dis]
        
        # getting next best action from neural network using TD3 policy based on current state
        action,episode_timesteps = policy.update(last_reward, curent_state1,current_state2,episode_timesteps,image_step,done)
        done = 0
               
        # If the car does not reach the distination within 1000 steps ,setting episode to done , so that it can start to train
        print("-----Done-----",done)
        if episode_timesteps > 500:
           done = 1 
           reward = 1                  
        
        
        rotation = action        
        #rotate the based the output of  Deep neural network model
        self.car.move(rotation)
          

    # check if car has reached the walls border. if so resetting car to centre
        if self.car.x < 25:
            self.car.x = 25
            print(".....Reached left Border.............")
            last_reward = -25
            #done = 1
        if self.car.x > self.width - 50:
            # self.car.x = self.width - 50
            self.car.x = self.width - 50
            last_reward = -20
            print(".....Reached right Border.............")
            #done = 1
        if self.car.y < 25:
            self.car.y = 25
            last_reward = -20
            print(".....Reached top Border.............")
            #done = 1
        if self.car.y > self.height - 25:
            self.car.y = self.height - 25
            last_reward = -50
            print(".....Reached bottom Border.............")
            #done = 1

            if distance < 25:
                done=1
                print("--------Reached Destination ---------")
                print("||||||car location|||||", self.car.x,self.car.y)
                last_reward = 10
                if swap == 1:
                    print("----------Dest 1 REACHED----------")
                    goal_x = 1420
                    goal_y = 622
                    swap = 0
                else:
                    print("--------Dest 2 REACHED----------")
                    
                    goal_x = 143
                    goal_y = 214
                    print("Next destination:" , goal_x, goal_y)
                    swap = 1
            else:
              done=0
        last_distance = distance
        

    # check the location of car
        if sand[int(self.car.y),int(self.car.x)] > 0:
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            reward = -0.85 if distance < last_distance else -1
        else: # otherwise
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            reward = last_reward + 0.2 if last_reward >= 0.3 else 0.3 # if it was on road last time or not
            if last_reward >= -0.5: # punishment or reward for moving toward goal
                reward = last_reward - 0.5 if distance >= last_distance else last_reward + 0.5 
               
        last_reward = reward    

class CarApp(App):
    
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent



#Running the whole thing
if __name__ == '__main__':
    CarApp().run()
