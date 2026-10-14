from PIL import Image
from pylab import *
from numpy import *
from PCV.tools import imtools

# 添加中文字体支持（使用系统自带黑体，避免乱码）
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  # 修正字体路径

# 修正图像路径（使用实际的绝对路径）
im = array(Image.open(r"E:\计算机视觉\JPG.jpg").convert('L'))  # 替换成你的图像路径

im2, cdf = imtools.histeq(im)

figure()
subplot(2, 2, 1)
axis('off')
gray()
title(u'(a)原始图像', fontproperties=font)
imshow(im)

subplot(2, 2, 2)
axis('off')
title(u'(b)直方图均衡化后的图像', fontproperties=font)
imshow(im2)

subplot(2, 2, 3)
axis('off')
title(u'(c)原始直方图', fontproperties=font)
hist(im.flatten(), 128, density=True)

subplot(2, 2, 4)
axis('off')
title(u'(d)均衡化后的直方图', fontproperties=font)
hist(im2.flatten(), 128, density=True)

show()  # 补充显示图像的命令