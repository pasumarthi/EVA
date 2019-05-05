### What are Channels and Kernels (according to EVA)?

Kernels help in feature extractor of an image. It helps in detecting edges and gradients in the initial layers and Textures and patterns, part of image, Image and scenes eventually at later stages of layers. Generally, 3*3 Kernel size are used. But you can also use any kernel size say 5*5 or 7*7 or 11*11.

Channel represents a feature. This feature can be like edges, gradients, parts  of object.  Multiple channel represents multiple features. 16 channel means 16 different features For grey image there will be only one Channel. RGB is represented by default 3 channels. RGB can be represented in multiple channels. That means these 3 colors can be represented in multiple channel. That is mixing these 3 colors can give different colors nothing but different channels. Each of the channels in each pixel represents the intensity of each color that constitute that pixel.

### Why should we only (well mostly) use 3x3 Kernels?

	It takes less parameters for example I can use three 3*3 kernels which 27 parameters
7*7 takes 49 parameters
From feature extraction perspective 7*7 is equivalent  to three 3*3 filers

	Nvidia have optimized their GPUs to 3*3 filters

	Also it is always recommend to use odd combinations as it has central line that helps in detection of edges comparatively to even combination which does proper not have proper center line.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
We need to perform 99 times 3*3 operations <br/>
Please refer the attaached excel for convulation operation
