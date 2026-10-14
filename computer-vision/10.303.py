from pylab import *
from PIL import Image
from PCV.localdescriptors import harris
from PCV.tools.imtools import imresize

im1 = array(Image.open(r"E:\计算机视觉\JPG.jpg").convert("L"))
im2 = array(Image.open(r"E:\计算机视觉\JPG.jpg").convert("L")) 


im1 = imresize(im1, (int(im1.shape[1]/2), int(im1.shape[0]/2)))
im2 = imresize(im2, (int(im2.shape[1]/2), int(im2.shape[0]/2)))

wid = 5
harrisim = harris.compute_harris_response(im1, 5)
filtered_coords1 = harris.get_harris_points(harrisim, wid+1)
d1 = harris.get_descriptors(im1, filtered_coords1, wid)

harrisim = harris.compute_harris_response(im2, 5)
filtered_coords2 = harris.get_harris_points(harrisim, wid+1)
d2 = harris.get_descriptors(im2, filtered_coords2, wid)

print('starting matching')
matches = harris.match_twosided(d1, d2)

figure()
gray()
harris.plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
show()