from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open("E:\计算机视觉\photo\JPG.jpg").convert('L'))
print(int(im.min()), int(im.max()))

im2 = 255 - im  
print(int(im2.min()), int(im2.max()))

im3 = (100.0/255) * im + 100 
print(int(im3.min()), int(im3.max()))

im4 = 255.0 * (im/255.0)**2 
print(int(im4.min()), int(im4.max()))

figure()
gray()
subplot(1, 3, 1)
imshow(im2)
axis('off')
title(r'(a) $f(x)=255-x$')

subplot(1, 3, 2)
imshow(im3)
axis('off')
title(r'(b) $f(x)=\frac{100}{255}x+100$')

subplot(1, 3, 3)
imshow(im4)
axis('off')
title(r'(c) $f(x)=255\left(\frac{x}{255}\right)^2$')
show()