% 读取
dicomFile ="E:\杂物\新建文件夹\series-000001\image-000001.dcm";  
img = dicomread(dicomFile);  % 读取DICOM图像;

% 显示原始图像
figure, subplot(1,2,1);
imshow(img, []);  % []自动缩放灰度范围以适配显示
title('原始医学图像');

img_uint8 = im2uint8(img); 
subplot(1,2,2);
imhist(img_uint8);
title('原始图像灰度直方图');