1.	How many layers
By default in neural network there will be 3 layers minimum. Input layer, hidden layer and output layer. The number of hidden layers is something one decides depends on the complexity of images in the data set, and type of hyper parameters we use in tuning to reach the local minima.  There is no free lunch. It is more of trial and error keeping end goal in mind. The main goal is generally to learn from the available training data as much as in a generalized fashion possible (without overfitting and underfitting) so that it helps when new images  are asked to recognize  or classify or predict .Generally first few layers learn simple features and later layers  complex features by merging the features learned in initial layers
2.	MaxPooling
Maxpool does not extract features. It picks max value of convolved features of the image. It helps in extracting sharp features. And reduce number of parameters. Improve computation. And takes care of translational and rotational invariance. Reduce overfitting. It is generally applied to non-overlapping regions of convolved images. Does not reduce the number of channels. 
1x1 Convolutions
It helps in carry forwarding only the required information and discard other information. It needs less number of parameters. It can use of existing channels to create complex channels. Less computation requirement for reduce number of channels. We can use 1*1 to increase number of channel based on the need, This is helpful as this acts like filtering. Imaging the last few layers seeing only the dog, instead of the dog sitting on the sofa, the background walls, painting on the wall, shoes on the floor and so on. If the network can filter out unnecessary pixels, later layers can focus on describing our classes more, instead of defining the whole image. 


3.	3x3 Convolutions
It helps in capturing spatial representations of the image.  It is always recommended to use odd combinations as it has central line that helps in detection of edges comparatively to even combination which does proper not have proper center line. Nvidia have optimized their GPUs to 3*3 filters. Also it takes less parameters when compared to 5*5 and 7*7 filters. Convolution with 3x3 kernel on 4x4 image will have a output channel with  a resolution of 2x2. We essentially lose 2 pixels in x as well as the y-axis. 

4.	Receptive Field
Every neuron in a convolutional layer represents the response of a filter applied to the previous layer. Receptive field can be defined as the ability of each neuron to see the part of image in previous layer . There is local receptive and Global receptive field. Local receptive field is the ability of each neuron to see the previous layer. Global receptive field part of input from the current layer. Local receptive field size is generally the size of filter.
Image size	filter	Local receptive field	Global Receptive field
28*28	3*3	3*3	3*3
	3*3	3*3	5*5
	3*3	3*3	7*7
	3*3	3*3	11*11
If the size of image is same as size of object, then global receptive field would be same as size of image. It is important for the network to see the whole image before it can predict exactly what the image is all about. 


5.	SoftMax,
SoftMax is going to create distance between the predicted number. If the numbers are near, it will pull apart. It is probability like. It is not probability. From GDPR perspective if someone asks we should not talk of SoftMax, we must show actual probability before SoftMax. The SoftMax function is often used in the final layer of a neural network-based classifier. 
6.	Learning Rate
Learning rate is used in back propagation. It helps reduce the loss by reducing the weights of the network with respect the loss gradient. The higher the learning faster the network will try to converge. But sometime if it is too high it will diverge rather will overshoot the minimum.  The slower the learning late the slower it will converge and may take more time to converge and more GPU cycles. Learning rate should be changed with change in Batch size.

7.	Kernels and how do we decide the number of kernels?
Initial layers generally less number of kernel layers and it finds edge and gradients . Later layers the number of kernel layers should be high to capture more detailed view of the images. Number of kernels should be reduced to overcome fitting. Higher kernel size will high computation cost. Ideally 3*3 kernel size is recommended. Number kernels should be higher than number of classes.  Higher number of convolutional kernels creates a higher number of channels/feature. Number of kernel in previous layers= Number of channels in current layer. Kernel can also be called feature extractor or filter. More kernels help in more granular details. However when you use more kernels than required it leads to overfitting.

8.	Batch Normalization,
The BatchNorm normalizes the net activations so that they have zero mean and unit variance. It does internal covariate shift which simply means change in the input distribution. That helps in faster convergence speed and better generalization performance. If you are using mini batch, every time mean, and variance calculated will be different for each mini batch. This adds bit of noise as every mini batch will have its own mean and variance. Hence during testing average of all the running mean and variance is considered. This helps reduce overfitting 
But not before first convolution. Batch Normalization should not be used before ReLU since the nonnegative responses of ReLU will make the weight layer updated in a suboptimal way. 
•	We can use higher learning rates because batch normalization makes sure that there’s no activation that’s gone high or low. And by that, things that previously couldn’t get to train, it will start to train.
•	It reduces overfitting because it has a slight regularization effect. Like dropout, it adds some noise to each hidden layer’s activations. Therefore, if we use batch normalization, we will use less dropout, which is a good thing because we are not going to lose a lot of information. However, we should not depend only on batch normalization for regularization; we should better use it together with dropout.



9.	Image Normalization,
It helps to reduce the values between 0 and 1. If the dataset contains images of which some are fully dark and which are fully white and model is trained with this kind of data set. It will have a challenge when recognition images with medium colors. So it is important to normalize the pixels in both the kind of images in the dataset so that it helps in recognition of all kinds of image irrespective of dark, light bright. 
10.	Position of MaxPooling,
MaxPooling should be applied after 2-3 convolution layers when sufficient information like edges, gradients , patters, parts of images are captured so that sharp features can be captured . Position of Maxpool may vary after some number of convolution layers depending on the complexity and type of the images. One should not jump to max pooling early without proper convolution layers. It should be introduced periodical in between the convolution layers after enough information has been extracted. It should be avoided in last layer. It leads to flattening 

11.	Concept of Transition Layers,
The transition layers consist of MaxPool or 1*1 layer or Batch Normalization. It transitions without extracting any features. No filter operations are done in transition layer.
12.	Position of Transition Layer,
Transition Layer is positioned after functional block which contains only Convolution layers.
CONVOLUTION BLOCK 2 BEGINS
         195x195x32     |(3x3x32)x64        | 193x193x64      RF of 24x24
         193x193x64     |(3x3x64)x128      | 191x191x128    RF of 26x26
         191x191x128   |(3x3x128)x256   | 189x189x256    RF of 28x28
         189x189x256   |(3x3x256)x512   | 187x187x512    RF of 30x30
CONVOLUTION BLOCK 2 ENDS
 
TRANSITION BLOCK 2 BEGINS
         MAXPOOLING(2x2)
         93x93x512 | (1x1x512)x32   | 93x93x32 RF of 60x60
TRANSITION BLOCK 2 ENDS


13.	Number of Epochs and when to increase them,
The greater number of epochs help to learn the images much better. It helps to increase the training accuracy. But it should not be increased so much that will lead to overfitting. Also, if it is too less also the network won’t be able to learn all the features of training images. The number of epochs should stop if training accuracy start decreasing.

14.	DropOut: It drops the percentage of kernels randomly in each iteration mentioned in the dropout(value). In turn each iteration, it drops different set of nodes and this results in a different set of outputs. So, can’t rely on one node or feature as we don’t know which one will be dropped hence weights should be spread out. (Hence ideally should be used after Batch Normalization) Rather helps in generalization of the training data set. In short, it helps in regularization. It is generally used when the network is huge to introduce randomness. It should not be used before prediction layer. This helps to reduce overfitting. Generally, dropout should be more (like 0.5) when you have large number of parameters in layer and more when less parameters (1 or 0.8). The input layer can also contain dropout generally it is 1 or 0.9)


15.	When do we introduce DropOut, or when do we know we have some overfitting
It should be introduced when we observe that the test accuracy is less than training accuracy. Rather difference between training and test accuracy is more with validation accuracy less than training accuracy. This helps in reduce number of kernels. Also when we have large number of parameters. 
Overfitting is not a problem when your test data is same a training. If you are training on Chinese people and plan to test on Chinese only. then overfitting is good to have.
16.	The distance of MaxPooling from Prediction,
It should not be layer before the prediction layer. It should be position minimum of 2 convolution layers before prediction. Otherwise, it would not help in generalization of the features.
17.	The distance of Batch Normalization from Prediction,
This will reduce the training accuracy if it is positioned before prediction layers as it normalizes the data.
18.	When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
When we reach the required receptive field, we can go with large kernel. In short size of image is less than object provided you are not using padding.

19.	How do we know our network is not going well, comparatively, very early
If the initial first 2 iterations show comparatively low training accuracy
If the difference between training and validation accuracy is too high
If the difference between training and validation accuracy is not reducing



20.	Batch Size, and effects of batch size 
If Batch size is more, then the time taken to run for an epoch is less. If your hardware supports you can make Batch size equal to total number of images.
Number of iterations= Total number of images/Batch size
Batch size should be more than number of classes. The ideal batch size would depend on the size and variants of training data set. Larger Batch size means larger learning rate.Backward propagation happens after each batch run. It does single upgrade of weights of all the image in a batch during back propagation. Batch size during testing should not matter.
. 
21.	When to add validation checks
It should be added to monitor overfitting and underfitting during training of the datasets. That would help to tune hyper parameters to reduce the diff between training and testing accuracy. And improve training and validation accuracy.
22.	LR schedule and concept behind it
Learning rate is often useful to reduce learning rate as the training progresses . It schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule. Initial layers it is generally high and is lower at final layers where it tries to stabilize
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
Adam vs SGD
Both Adam and SGD are optimization algorithms. They help to reduce the loss. Loss is calculated difference actual and observed value. Observed values are calculated by weight and bias which are updated by learning parameters
SGD takes one sample per step in case of mini batch or subset of data takes one sample for batch. It is quite useful when we large of data and lots of parameters.
Adam is an adaptive learning rate optimization algorithm that’s been designed specifically for training deep neural networks. the algorithms leverage the power of adaptive learning rates methods to find individual learning rates for each parameter.



