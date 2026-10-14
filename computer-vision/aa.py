from PIL import Image
import pylab 

image_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.jpg"
save_path = r"C:\Users\ahdezneytiry-boop1\Desktop\图片\M6S98lbBcBa63V7.png"

im = Image.open(image_path)
im.save(save_path, "png")

pylab.imshow(im) 
pylab.axis("off")  
pylab.show()     