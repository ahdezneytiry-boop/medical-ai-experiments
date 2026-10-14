from PCV.localdescriptors.harris import *
from PIL import Image
from numpy import array

if __name__=='__main__':
    im = array(Image.open(r"E:\计算机视觉\JPG.jpg").convert('L'))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim, 50)
    plot_harris_points(im, filtered_coords)