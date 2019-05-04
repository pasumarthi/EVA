•	What are Channels and Kernels (according to EVA)?
Kernels help in feature extractor of an image. It helps in detecting edges and gradients in the initial layers and Textures and patterns, part of image, Image and scenes eventually at later stages of layers. Generally, 3*3 Kernel size are used. But you can also use any kernel size say 5*5 or 7*7 or 11*11.

Channel represents a feature. This feature can be like edges, gradients, parts  of object.  Multiple channel represents multiple features. 16 channel means 16 different features For grey image there will be only one Channel. RGB is represented by default 3 channels. RGB can be represented in multiple channels. That means these 3 colors can be represented in multiple channel. That is mixing these 3 colors can give different colors nothing but different channels. Each of the channels in each pixel represents the intensity of each color that constitute that pixel. 
Why should we only (well mostly) use 3x3 Kernels?
	It takes less parameters for example I can use three 3*3 kernels which 27 parameters
7*7 takes 49 parameters
From feature extraction perspective 7*7 is equivalent  to three 3*3 filers

	Nvidia have optimized their GPUs to 3*3 filters

	Also it is always recommend to use odd combinations as it has central line that helps in detection of edges comparatively to even combination which does proper not have proper center line.

•	How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
We need to perform 99 times 3*3 operations
199*199
197*197
195*195
193*193
191*191
189*189
187*187
185*183
181*181
179*179
177 *177
175*175
173*173
171*171
169*169
167*167
165*165
163*163
161*161
159*159
157*157
155 *155
153*153
151*151
149*149
147*147
145*145
143*143
141*141
139*139
137*137
133*133
131*131
129*129
127*127
125*125
123*123
121*121
119*119
117*117
115*115
113*113
111*111
109*109
107*107
105*105
103*103
101*101
99*99
97*97
95*95
93*93
91*91
89*89
87*87
85*85
83*83
81*81
79*79
77*77
75*75
73*73
71*71
69*69
67*67
65*65
63*63
61*61
59*59
57*57
55*55
53*53
51*51
49*49
47*47
45*45
43*43
41*41
39*39
37*37
35*35
33*33
31*31
29*29
27*27
25*25
23*23
21*21
19*19
17*17
