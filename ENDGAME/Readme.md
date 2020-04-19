
The initial challenge was to git rid of  sensors and use  the car image as the location . 
Reading the image state after cropping it.
Had a challenge in new_state shape and fixed 
Got the angle(actions) from the DQN policy to be rotated. Which I used for movement of the car.
Then ran the with  DQN . And was able to see the movement of car moving from source to destination

Once got confidence that car movement was being captured then tried to port the code to TD3
As it  was in the car is in continuous space , wanted to removed 3 actions [-5,0,5] .Then ,used random.uniform to get random values between [-5,5]

As TD3 had 6 networks, the debugging became very slow .  And next  challenge how to pass the image and state and action to critic.  This I solved by flatten the image state and concatenated with actions 
 
Introduced the done to complete of episode and in line with Replay buffer of TD3

To achieve random movement without training , the update method has to be modified to get the count of car movements
Also when car reached the border or the edges of canvas repositioned  to center of canvas. Also tried setting to random location but not much use improvement 
Modified the code to get per episode rewards  and time_stamps 

Tried with reward function combinations and learning rates.

Car was rotating certain times at the same place within sand, then introduced if it does not complete the episode in 2500 steps, then repositioned  to center of canvas
Had to try different iterations and decided to go with 5000 steps

To achieve the same the policy.update has to be modified to capture number of steps along with action


Tried by modifying the learning rate to check if proper actions/rotation and also the different hyper parameters of TD3
