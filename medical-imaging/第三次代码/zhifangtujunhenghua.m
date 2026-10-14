% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 计算原始图像的直方图
% 直方图均衡化
J = histeq(I); 

% 四图对比显示
figure('Name', '直方图均衡化对比');

% 1. 原始图像
subplot(2,2,1);
imshow(I, []);  % 自动适配灰度范围显示
title('原始医学影像');

% 2. 原始图像直方图
subplot(2,2,2);
imhist(I);
title('原始图像直方图');

% 3. 均衡化后图像
subplot(2,2,3);
imshow(J, []);
title('直方图均衡化后图像');

% 4. 均衡化后直方图
subplot(2,2,4);
imhist(J);
title('均衡化后直方图');