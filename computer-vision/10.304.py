import cv2
import numpy as np

# 核心修复：使用纯英文路径（E盘根目录，无任何中文）
# 操作步骤：
# 1. 把你的 JPG 图片复制到 E 盘根目录（打开“此电脑”→双击 E 盘，直接粘贴图片）
# 2. 把图片文件名改为简单英文（如 test.jpg，避免中文/特殊字符）
filename = r"E:\JPG.jpg"  # 纯英文路径+文件名，绝对不会乱码

# 读取图片（添加异常处理）
img = cv2.imread(filename)
if img is None:
    print(f"错误：无法读取图片！请按以下步骤检查：")
    print(f"1. 确认图片已放在 E 盘根目录，文件名为 test.jpg")
    print(f"2. 图片格式是 JPG（后缀为 .jpg，不是 .png/.jpeg 等）")
    print(f"3. 图片未损坏（可以用画图打开验证）")
    exit()

# 角点检测核心代码
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 6, 3, 0.04)
img[dst > 0.01*dst.max()] = [0, 0, 255]  # 角点标记为红色

# 显示结果
cv2.imshow('Corner Detection', img)
if cv2.waitKey(0) & 0xff == 27:  # 按 ESC 键关闭窗口
    cv2.destroyAllWindows()