from PIL import Image,ImageFile
import CV2
import os, sys
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = "C:/rough1/abcd1/"
dirs = os.listdir( path )
print(len(os.listdir("C:/rough1/abcd1")))
for item in dirs:
#if os.path.isfile(path+item):
    im = Image.open(path+item)
    print(im.size,im.mode)
    f, e = os.path.splitext(path+item);print(f)
    imResize = im.resize((400,400), Image.ANTIALIAS)
    imResize.save(f + ' resized.jpg', 'JPEG', quality=90)
    


