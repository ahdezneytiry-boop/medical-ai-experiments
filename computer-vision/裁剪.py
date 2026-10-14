from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

image_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.jpg" 

origin_im = Image.open(image_path) 

plt.subplot(121)
plt.title("(a) ")
plt.axis("off") 
plt.imshow(origin_im)

crop_box = (100, 500, 400, 700) 
plt.subplot(122) 
plt.title(f"(b) ")
plt.axis("off")
plt.imshow(origin_im.crop(crop_box)) 
plt.tight_layout()
plt.show()