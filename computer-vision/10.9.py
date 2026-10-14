from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

try:
    im = Image.open("E:\计算机视觉\photo\JPG.jpg") 
    r, g, b = im.split()
    
    plt.figure()
    plt.subplot(121)
    plt.title(u'(a)原图像', fontproperties=font)
    plt.axis('off')
    plt.imshow(im)
    
    plt.subplot(122)
    plt.title(u'(b)图像直方图', fontproperties=font)
    # 绘制RGB三色直方图
    plt.hist(np.array(r).flatten(), bins=256, density=True, 
             facecolor='yellowgreen', edgecolor='yellowgreen', alpha=0.7)
    plt.hist(np.array(g).flatten(), bins=256, density=True, 
             facecolor='pink', edgecolor='pink', alpha=0.7)
    plt.hist(np.array(b).flatten(), bins=256, density=True, 
             facecolor='deepskyblue', edgecolor='deepskyblue', alpha=0.7)
    
    plt.show()
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"发生错误: {e}")