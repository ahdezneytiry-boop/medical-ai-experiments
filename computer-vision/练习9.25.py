# 使用matplotlib正确连接四点形成矩形
from PIL import Image
from pylab import *

# 读取图像
im = array(Image.open("F:\计算机视觉\photo\JPG.jpg"))

# 显示图像
imshow(im)

# 四个点的坐标列表（按顺时针顺序排列的矩形四个角）
# 顺序：左上 → 右上 → 右下 → 左下 → 左上（闭合）
x = [100, 400, 400, 100]  # 正确的x坐标顺序
y = [200, 200, 500, 500]  # 正确的y坐标顺序

# 用红色叉型标出所有点，增大标记大小
plot(x, y, 'rx', markersize=10)

# 依次连接四个点并闭合形成矩形
plot(x + [x[0]], y + [y[0]], 'b-', linewidth=2)


show()
