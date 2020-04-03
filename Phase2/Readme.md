Step-2:



<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step2.jpg" width="350" >
 </p>

Step-3:



<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step3.jpg" width="350" >
 </p>




Step-4:

Load samples from replay buffer in batches, for state, next_state,action, reward and done(flag to show status of episode)
Load these samples on to the device(cpu or gpu).

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step4.jpg" width="350" >
 </p>

Step-5:

Using next_state(s') calculate next action(a') with the actor target model as its forward pass.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step5.jpg" width="350" >
 </p>
 
 Step-6:

Add noise to the next action(a') from a normal distribution.
Clip the a' values to main a range of values with in the normal distribution.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step6.jpg" width="350" >
 </p>
 
 Step-7:

Using both s' and a' with added noise calculate Q1t and Q2t values for both critic targets as its forward pass.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step7.jpg" width="350" >
 </p>
 
Step-8:

Of the two critic target values, take the minimum value.
We take the minimum value to avoid the over estimation of the Q-value and overcome the problem of variance.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step8.jpg" width="350" >
 </p>

Step-9:

Target Q-values(min of two critic targets), is discounted from next_state(s') to state(s) and added to current state-action reward.(Bellmen eq)
When the terminal state of the episode is reached, the next state does not exist. To show this we use flag 'done', with done being 1 if the state is the terminal state 0 otherwise. If we reached end of episode done=1, and right side of plus symbol becomes zero.
detatch in pytorch

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step9.jpg" width="350" >
 </p>
 
Step-10:

Predict state(s) Q-values with critic-model using state(s) and corresponding action(a) values taken form replay buffer in forward pass.
<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step10.jpg" width="350" >
 </p>

Step-11:

calculate mse-loss using the optimal target Q-value as reference, with respect to corresponding Q-values(Q1,Q2) of critic-model.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step11.jpg" width="350" >
 </p>

Step-12:

Using the mse-loss from the Step-11, backpropagate to critic-models.
<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step12.jpg" width="350" >
 </p>


Step-13:

We use critic-model(Q1) to update the actor model, by applying graiant ascent on the Q-value approximation function.
Rather than doing gradinat ascent on Q-value, we are doing gradint descent on negative of Q-value.
For every two critic-model updtes, we update the actor-model once. This is to delay the actor-model update, to avaoid instant updates to critics for stability.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step13.jpg" width="350" >
 </p>


Step-14:

Polyak avg for critic-target models, to get stable target Q-values.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step14.jpg" width="350" >
 </p>


Step-15:

Polyak avg for actor-target models, to get stable target Q-values.

<p align="center">
  <img src="https://github.com/pasumarthi/EVA/blob/master/Phase2/images/Step15.jpg" width="350" >
 </p>
