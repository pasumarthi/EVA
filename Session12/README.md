The intial intialization very important. This helps in how gradients are calculated and weights are updated matters a lot. Keras uses Xavier golor init policy. pytorch uses kaiminghe intialization policy. Keras does not have Kaihment policy. Here we are using pytorch poilcy of intialization  but not using pytorch. Kaihment intialization policy seems much better than Xavier and otehr policies.These small differences matter a  lot.


Here classses are written in pytorch fashoin. tensor2.0 has started doing this way

Conv2DBN 
the init functions has definations.
the call functions stich them together

Conv2DBN class name as it has Batch normalization.
 give cout channels
cnumber of filters and intialize with pytroch kaiman intializer

When Conv2DBN function with input layer is called 
input goes out to conv layer -->deoput layer-->batch norma --> relu activation layer

In Resnet block when is call self.conv2DBN all the 3 above are called

pool is adding part
if pool is call then max pooling happens
if res=true then resnet block is called

 h = h + self.res2(self.res1(h)) number of skip connection


In David net by default it is called 64 channels
Maxpooiling is used to improve accuracy insteas of &*7 with stride of 2
And then Conv block is called
 Resnet block with with 128 channels and max pooling with residual true 
When res is true skip connection is used
Residual block with 256 channels and max pooling without residual connection 
If Resnet block with with 512 channels and max pooling with residual true 
And then Globalmaxpooling is done and denselayer connect to final 10 outputs.

Down load cifar10 images ,calulate mean and standard deviation
And padded with 4*4 
And normalized padded trained images 



One cycle policy 


Do range test and pick max lr which has lower losss and then pick i/5th of the leraning rate
We go from lower learning rate to higher learning rate in step 1 and back to lower learning rate in step 2. 
the interploation function creates the triangle for One Cycle LR 

It divides the lr by 5 and goes to max  in the number of steps [0, (EPOCHS+1)//5, EPOCHS] and then come down [0, LEARNING_RATE, 0]
Gloabal learning take care of save and infuse it back 

Image augmentation
The padded images (40*40) are randomly cropped 32* 32 and randomly fliped left and right.that is what becomes x and y reamin y


Some of the keras functions are wtitten to dynamically load learning rate.

https://github.com/bstriner/keras-tqdm
TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
https://www.tensorflow.org/api_docs/python/tf/GradientTape
GradientTapes can be nested to compute higher-order derivatives.


In the run you you could see the learning rate increasing from 0.08 to to 0.4 and tehn getting down to 0.0 
in 23 epcohs we could reach 93.01% accuracy.



