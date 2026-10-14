from PIL import Image
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取并处理图像
image_path = r"E:\计算机视觉\9.18\M6S98lbBcBa63V7.jpg"
pil_im = Image.open(image_path)

# 裁剪
box = (100, 100, 400, 400)
region = pil_im.crop(box)

# 粘贴
pil_im_paste = Image.open(image_path) # 使用副本进行粘贴
pil_im_paste.paste(region, box)

# 缩放后旋转
im_resize = pil_im_paste.resize((128, 128))
im_rotated = im_resize.rotate(45, expand=True)

# 2. 创建一个包含3个子图的画布
#    设置figsize使整体布局更美观
plt.figure(figsize=(15, 5))

# 3. 绘制第一张图：裁剪区域
plt.subplot(1, 3, 1)  # (行数, 列数, 当前子图索引)
plt.title('裁剪后的复制区域')
plt.axis('off')
plt.imshow(region)

# 4. 绘制第二张图：粘贴后的图像
plt.subplot(1, 3, 2)
plt.title('粘贴后的图像')
plt.axis('off')
plt.imshow(pil_im_paste)

# 5. 绘制第三张图：缩放+旋转后的图像
plt.subplot(1, 3, 3)
plt.title('缩放(128x128)+旋转(45度)')
plt.axis('off')
plt.imshow(im_rotated)

# 6. 自动调整子图间距，避免重叠
plt.tight_layout()

# 7. 显示最终的组合图像
plt.show()

# (可选) 保存最终生成的组合图像
# plt.savefig(r"E:\计算机视觉\9.18\combined_results.png", dpi=300)

# (可选) 保存最终处理的单张图像
# im_rotated.save(r"E:\计算机视觉\9.18\final_rotated_image.jpg")