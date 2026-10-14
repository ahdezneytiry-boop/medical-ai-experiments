# contour 与 hist
# 绘制轮廓要将图像先灰度化
from PIL import Image
from pylab import *

# 打开图像并转换为灰度图，再转换为数组
im = array(Image.open("E:\计算机视觉\photo\JPG.jpg").convert('L'))
figure()  # 创建新的图像窗口

gray()  # 设置为灰度显示模式
# 绘制图像轮廓，origin='image' 使轮廓起始位置与图像左上角对齐
contour(im, origin='image')
axis('equal')  # 让坐标轴刻度均匀分布（x、y方向缩放比例一致）

# 新建图像窗口，绘制直方图（展平数组+256个区间）
figure()
hist(im.ravel(), 256)

# 新建图像窗口，绘制直方图（展平数组+128个区间，灰度值跨度为 256/128=2）
# hist的第二个参数指定小区间的个数为128，即每个小区间灰度值跨度为2
figure()
hist(im.flatten(), 128)

show()  # 显示所有图像窗口