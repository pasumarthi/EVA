import scipy.ndimage
from PIL import Image as PILImage
import numpy as np
from matplotlib import pyplot as plt

sand = np.asarray(PILImage.open("./images/MASK1.png").convert('L'))/255
car_x, car_y = 737,256
angle = 40

def _subimage(sand, car_x, car_y, angle, crop_size=40): 
  # function takes sand image as input and center positions of car as input
  # returns an np array of angled cutout: shape (40,40,1)
  pad = crop_size*2
  #pad for safety
  crop1 = np.pad(sand, pad_width=pad, mode='constant', constant_values = 1)
  centerx = car_x + pad
  centery = car_y + pad

  #first small crop
  startx, starty = int(centerx-(crop_size)), int(centery-(crop_size))
  crop1 = crop1[starty:starty+crop_size*2, startx:startx+crop_size*2]

  #rotate
  crop1 = scipy.ndimage.rotate(crop1, -angle, mode='constant', cval=1.0, reshape=False, prefilter=False)
  #again final crop
  startx, starty = int(crop1.shape[0]//2-crop_size//2), int(crop1.shape[0]//2-crop_size//2)
  return crop1[starty:starty+crop_size, startx:startx+crop_size].reshape(crop_size, crop_size, 1)

#testing our cutout
_cutout = _subimage(sand, car_x, car_y, angle, crop_size=40)
plt.imshow(_cutout.reshape(40, 40))
plt.show()
