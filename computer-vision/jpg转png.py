# 1. 正确导入依赖库
from PIL import Image
import pylab  # 更规范的导入方式（避免*导入冗余内容）

# 2. 处理路径（原始字符串）
image_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.jpg"
save_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.png"

# 3. 打开并保存图像
im = Image.open(image_path)
im.save(save_path, "png")

# 4. 用pylab显示图像（先加载再显示）
pylab.imshow(im)  # 关键：将图像加载到pylab窗口
pylab.axis("off")  # 可选：隐藏坐标轴
pylab.show()       # 显示窗口