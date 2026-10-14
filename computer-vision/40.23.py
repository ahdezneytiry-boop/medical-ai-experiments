# 图像模糊
# Scipy库
from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open(r"E:\计算机视觉\JPG.jpg").convert('L'))
# filters.gaussian_filter第二个参数是标准差
im2 = filters.gaussian_filter(im, 9)
show(im2)