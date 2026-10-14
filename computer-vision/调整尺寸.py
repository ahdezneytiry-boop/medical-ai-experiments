from PIL import Image
from pylab import *

im = Image.open("E:\计算机视觉\9.18\M6S98lbBcBa63V7.jpg", "r")
im.show()

im_resize = im.resize((200,200))
im_resize.show()

im_resize_box = im.resize((100,100), box = (0,0,956,956))
im_resize_box.show()