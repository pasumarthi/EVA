# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

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
from ai1 import TD3

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

# Getting our AI, which we call "policy", and that contains our neural network that represents our Q-function
policy = TD3((1,40,40),1,5)


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
    swap = 0
    done=0
    image_step=0

# first destination 
    goal_x = 1420
    goal_y = 622


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
        self.car.velocity = Vector(6, 0)

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
        

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

    #getting car image at the current location.

        image = sand[int(self.car.x)-20:int(self.car.x)+20, int(self.car.y)-20:int(self.car.y)+20]
    
    # calculating number of steps of image so that it can be used for random movement
        image_step += 1

        last_signal = image

    # getting next best action from neural network using TD3 policy based on current state
        action,episode_timesteps = policy.update(last_reward, last_signal,episode_timesteps,image_step,done)

        done = 0
        
        rotation = action
        #print(".........amount of rotation",rotation)
        #rotate the based the out of Deep neural network model
        self.car.move(rotation)

    # calculate the distance from destaination after current move of car
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)


    # If the car does not reach the distination within 2500 steps resetting the car  to centre of canvas
        if episode_timesteps > 5000:
            self.car.x = self.width/2
            self.car.y = self.height/2
            #done = 1

    # check the location of car
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -15
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.25
            #print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.3
           
    # check if car has reached the walls border. if so resetting car to centre
        if self.car.x < 20:
            self.car.x = self.width/2
            self.car.y = self.height/2
           # self.car.x=20
            print(".....Reached Border.............")
            last_reward = -30
            done = 1
        if self.car.x > self.width - 20:
            # self.car.x = self.width - 50
            self.car.x = self.width/2
            self.car.y = self.height/2
            last_reward = -30
            print(".....Reached Border.............")
            done = 1
        if self.car.y < 20:
            # self.car.y = 50
            self.car.x = self.width/2
            self.car.y = self.height/2
            #self.car.y=20
            last_reward = -30
            print(".....Reached Border.............")
            done = 1
        if self.car.y > self.height - 20:
            # self.car.y = self.height - 50
            self.car.x = self.width/2
            self.car.y = self.height/2
            last_reward = -30
            print(".....Reached Border.............")
            done = 1

       
        if distance < 25:
            done = 1
            last_reward = 40
            self.car.x = self.width/2
            self.car.y = self.height/2

               
            if swap == 5:
                print ("Reached destination goal_x = 143, goal_y = 214")
                goal_x = 1140
                goal_y = 480
                swap = 4
            elif swap == 4:
                print ("Reached destination goal_x = 1140, goal_y = 480")
                goal_x = 500
                goal_y = 624
                swap = 3
            elif swap == 3:
                print ("Reached destination goal_x = 500, goal_y = 624")
                goal_x = 1100
                goal_y = 310
                swap = 2
            elif swap == 2:
                print ("Reached destination goal_x = 1100, goal_y = 310")
                goal_x = 618
                goal_y = 42
                swap = 1
            else: 
                print ("Reached destination goal_x = 1420, goal_y = 622")
                goal_x = 143
                goal_y = 214
                swap = 5
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving policy...")
        policy.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved policy...")
        policy.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
