from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import histogram, interp

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def histeq(im, bins=256):
    im = im.astype(np.float32)
    im_flat = im.flatten()
    imhist, bins = histogram(im_flat, bins)
    cdf = imhist.cumsum()
    cdf = cdf / cdf[-1] if cdf[-1] != 0 else cdf
    cdf = 255 * cdf
    im2_flat = interp(im_flat, bins[:-1], cdf)
    im2 = im2_flat.reshape(im.shape).astype(np.uint8)
    return im2, cdf

if __name__ == "__main__":
    img_path = r"E:\计算机视觉\JPG.jpg"
    try:
        with Image.open(img_path) as img:
            im_gray = img.convert('L')
            im = np.array(im_gray)
    except FileNotFoundError:
        print(f"错误：未找到图像文件，请检查路径是否正确：{img_path}")
        exit()
    except Exception as e:
        print(f"图像读取失败：{e}")
        exit()
    
    im_eq, cdf = histeq(im, 256)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title("原始灰度图像")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(im_eq, cmap='gray')
    plt.title("直方图均衡化后的图像")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.hist(im.flatten(), bins=256, range=(0, 255), density=True, color='blue', alpha=0.7)
    plt.title("原始图像的灰度直方图")
    plt.xlabel("灰度值（0-255）")
    plt.ylabel("概率密度")
    plt.xlim(0, 255)
    
    plt.subplot(2, 2, 4)
    plt.hist(im_eq.flatten(), bins=256, range=(0, 255), density=True, color='green', alpha=0.7)
    plt.title("均衡化后的灰度直方图")
    plt.xlabel("灰度值（0-255）")
    plt.ylabel("概率密度")
    plt.xlim(0, 255)
    
    plt.tight_layout()
    plt.show()