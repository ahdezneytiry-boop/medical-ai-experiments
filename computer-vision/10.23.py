import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def histeq(im, bins=256):
    imhist, bins = np.histogram(im.flatten(), bins)
    # 计算累计分布函数（CDF）
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # cdf[-1]是CDF的最大值（总像素数）
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape).astype(np.uint8), cdf
try:
    im = np.array(Image.open(r"E:\计算机视觉\JPG.jpg").convert('L'))  # 直接读取当前目录下的图片，或写完整路径如'./pic/kobe_mamba.jpg'
except FileNotFoundError:
    print("错误：未找到图片文件，请检查路径是否正确！")
    exit()

# 执行直方图均衡化
im2, cdf = histeq(im, 256)

# 显示均衡化后的图像和直方图
plt.figure(figsize=(10, 5))

# 显示均衡化后的图像
plt.subplot(121)
plt.imshow(im2, cmap='gray')
plt.title('均衡化后的图像')
plt.axis('off')
plt.subplot(122)
plt.hist(im2.flatten(), 256, color='gray')
plt.title('均衡化后的直方图')

plt.tight_layout()
plt.show()