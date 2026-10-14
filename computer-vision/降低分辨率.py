from PIL import Image
from pylab import *
im = Image.open("C:/Users/ahdezneytiry-boop1/Desktop/图片/M6S98lbBcBa63V7.jpg", "r")
subplot(121)
title('(a)')
axis("off")
imshow(im)

im.save("C:/Users/ahdezneytiry-boop1/Desktop/图片/M6S98lbBcBa63V7.png", 'png')
im.thumbnail((128,128), resample=Image.BICUBIC)
subplot(122)
title('(b)')
axis("off")
imshow(im)
show()