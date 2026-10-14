from PIL import Image
from pylab import *

# 打开图像（需确保dog.jpeg在当前路径）
im = Image.open("E:\计算机视觉\M6S98lbBcBa63V7.jpg")

# 1. 原始图像
subplot(331)
title('(a)')
axis('off')
imshow(im)

# 2. 左右翻转
Image_FLIP_LEFT_RIGHT = im.transpose(Image.FLIP_LEFT_RIGHT)
subplot(332)
title('(b)')
axis('off')
imshow(Image_FLIP_LEFT_RIGHT)

# 3. 上下翻转
Image_FLIP_TOP_BOTTOM = im.transpose(Image.FLIP_TOP_BOTTOM)
subplot(333)
title('(c)')
axis('off')
imshow(Image_FLIP_TOP_BOTTOM)

# 4. 旋转45度
Image_ROTATE_45 = im.rotate(45)
subplot(334)
title('(d)')
axis('off')
imshow(Image_ROTATE_45)

# 5. 旋转90度
Image_ROTATE_90 = im.transpose(Image.ROTATE_90)
subplot(335)
title('(e)')
axis('off')
imshow(Image_ROTATE_90)

# 6. 旋转180度
Image_ROTATE_180 = im.transpose(Image.ROTATE_180)
subplot(336)
title('(f)')
axis('off')
imshow(Image_ROTATE_180)

# 7. 旋转270度
Image_ROTATE_270 = im.transpose(Image.ROTATE_270)
subplot(337)
title('(g)')
axis('off')
imshow(Image_ROTATE_270)

# 8. 图像转置（类似矩阵转置）
Image_TRANSPOSE = im.transpose(Image.TRANSPOSE)
subplot(338)
title('(h)')
axis('off')
imshow(Image_TRANSPOSE)

# 9. 图像横向转置
Image_TRANSVERSE = im.transpose(Image.TRANSVERSE)
subplot(339)
title('(i)')
axis('off')
imshow(Image_TRANSVERSE)

# 显示所有子图
show()