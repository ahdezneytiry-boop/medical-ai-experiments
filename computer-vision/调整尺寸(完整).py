from PIL import Image

im = Image.open(r"E:\计算机视觉\9.18\M6S98lbBcBa63V7.jpg", "r")
print("原图真实尺寸（宽, 高）：", im.size)  

im.show()
im_resize = im.resize((200,200))
im_resize.show()

im_resize_box = im.resize((100,100), box = (300,250,700,600))  
im_resize_box.show()