from pylab import *
from PIL import Image
from PCV.localdescriptors import harris

# 读入图像
im = array(Image.open("E:\计算机视觉\JPG.jpg").convert('L'))
# 检测harris角点
harrisim = harris.compute_harris_response(im)
# Harris响应函数
harrisiml = 255 - harrisim
figure()
gray()
# 画出Harris响应图
subplot(141)
imshow(harrisiml)
print(harrisiml.shape)
axis('off')
axis('equal')
threshold = [0.01, 0.09, 0.5]
for i, thres in enumerate(threshold):
    filtered_coords = harris.get_harris_points(harrisim, 6, thres)
    subplot(1, 4, i + 2)
    imshow(im)
    print(im.shape)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
show()