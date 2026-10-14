from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --------------------------
# 1. 修复中文字体配置（关键！）
# 确保字体路径指向系统中存在的中文字体（如SimHei、微软雅黑等）
# 常见系统字体路径参考：
# - Windows: C:\Windows\Fonts\simhei.ttf（黑体）、C:\Windows\Fonts\msyh.ttc（微软雅黑）
# - Mac: /System/Library/Fonts/PingFang.ttc（苹方）
# 请根据自己的字体路径修改
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  # 替换为实际字体路径


# --------------------------
# 2. 自定义直方图均衡化函数（替代PCV的imtools.histeq）
def histeq(image, n_bins=256):
    """
    实现直方图均衡化
    参数:
        image: 输入灰度图像（二维数组）
        n_bins: 直方图分箱数（默认256）
    返回:
        equalized_image: 均衡化后的图像
        cdf: 累积分布函数
    """
    # 1. 计算图像直方图
    hist, bins = np.histogram(image.flatten(), n_bins, density=True)
    
    # 2. 计算累积分布函数（CDF）
    cdf = hist.cumsum()  # 累积求和
    cdf = 255 * cdf / cdf[-1]  # 归一化到0-255
    
    # 3. 用线性插值将CDF应用到图像，得到均衡化结果
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf)
    
    # 转换为与原图相同的形状和数据类型（uint8）
    return equalized_image.reshape(image.shape).astype(np.uint8), cdf


# --------------------------
# 3. 主程序：加载图像并处理
if __name__ == "__main__":
    # 请替换为你的图像路径（如.jpg、.png等）
    image_path = r"E:\计算机视觉\photo\JPG.jpg"  # 示例路径，需修改为实际图像
    
    # 打开图像并转换为灰度图
    try:
        im = Image.open(image_path).convert('L')  # 'L'表示灰度模式
        im = np.array(im)  # 转换为NumPy数组
    except Exception as e:
        print(f"图像加载失败：{e}")
        print("请检查图像路径是否正确！")
        exit()
    
    # 执行直方图均衡化（使用自定义函数）
    im2, cdf = histeq(im)
    
    # --------------------------
    # 4. 绘制结果（4个子图）
    plt.figure(figsize=(10, 8))
    
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.gray()
    plt.title(u'(a)原始图像', fontproperties=font)
    plt.imshow(im)
    
    # 均衡化后的图像
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.title(u'(b)直方图均衡化后的图像', fontproperties=font)
    plt.imshow(im2)
    
    # 原始图像的直方图
    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.title(u'(c)原始直方图', fontproperties=font)
    plt.hist(im.flatten(), 128, density=True)  # 128个分箱
    
    # 均衡化后的直方图
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.title(u'(d)均衡化后的直方图', fontproperties=font)
    plt.hist(im2.flatten(), 128, density=True)
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()