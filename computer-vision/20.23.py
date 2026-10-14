from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *

if __name__ == '__main__':
    # 读取图像并转换为灰度图
    im = array(Image.open("E:\计算机视觉\JPG.jpg").convert('L'))
    gray()  # 设置绘图颜色为灰度模式
    
    # 显示原始图像
    subplot(1, 4, 1)
    axis('off')
    title('Original Image')
    imshow(im)
    
    # 对不同高斯模糊参数进行循环处理
    for bi, blur in enumerate([2, 5, 10]):
        im2 = zeros(im.shape)
        # 应用高斯滤波
        im2 = filters.gaussian_filter(im, blur)
        # 转换为uint8类型以保证图像显示正常
        im2 = np.uint8(im2)
        imNum = str(blur)
        # 显示模糊后的图像
        subplot(1, 4, 2 + bi)
        axis('off')
        title('GaussVar = ' + imNum)
        imshow(im2)
    
    show()  # 显示所有子图