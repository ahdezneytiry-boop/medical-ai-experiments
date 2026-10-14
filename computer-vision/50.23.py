from PIL import Image
from numpy import *
from scipy.ndimage import filters
import matplotlib.pyplot as plt

# 彩色通道，三通道分别进行高斯滤波
im = array(Image.open("E:\计算机视觉\JPG.jpg"))
im2 = np.zeros((im.shape))
for i in arange(3):
    im2[:, :, i] = filters.gaussian_filter(im[:, :, i], 2)
# 转换为(0, 255)，否则imshow显示不出来
im2 = uint8(im2)
plt.figure(figsize=(5, 5), dpi=80)
plt.imshow(im2)
plt.axis('off')
plt.show()