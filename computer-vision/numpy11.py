from PIL import Image
from pylab import *

im = array(Image.open("E:\计算机视觉\photo\JPG.jpg"))
print(im.shape, im.dtype)
im = array(Image.open("E:\计算机视觉\photo\JPG.jpg").convert('L'), 'f')
print(im.shape, im.dtype)