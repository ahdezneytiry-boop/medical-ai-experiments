from PIL import Image  
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

image_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.jpg"

try:
    pil_im = Image.open(image_path)  
    pil_im.thumbnail((120, 120)) 
except FileNotFoundError:
    print(f"错误：未找到图片！请检查路径：{image_path}")
    exit()
except Exception as e:
    print(f"其他错误：{e}") 
    exit()

plt.title("缩略图") 
plt.xticks([]) 
plt.yticks([])  
plt.imshow(pil_im)  
plt.show() 