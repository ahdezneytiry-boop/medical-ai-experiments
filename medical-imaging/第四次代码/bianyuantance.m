% 边缘探测法
clear; clc; close all;

% 1. 读取并预处理图像
pngPath = 'C:\Users\ahdezneytiry-boop1\Desktop\018.png';
I1 = imread(pngPath);
I1_gray = rgb2gray(I1);
I1_gray = im2double(I1_gray);
% 高斯滤波+对比度增强
I_blur = imgaussfilt(I1_gray, 0.6);
I_enhance = adapthisteq(I_blur, 'ClipLimit', 0.03);

figure('Position', [100, 100, 1000, 600]);
subplot(2,2,1);
imshow(I1_gray);
title('(a) 原始图像');
subplot(2,2,2);
imshow(I_enhance);
title('(b) 增强后图像');

% 2. Canny算子
I_canny = edge(I_enhance, 'canny', [0.15, 0.35]);  % 适配高对比度病灶
% 边缘叠加+形态学闭合
I_canny_fill = imclose(I_canny, strel('disk', 2));
I_canny_seg = I1_gray;
I_canny_seg(I_canny_fill) = 1;

subplot(2,2,3);
imshow(I_canny);
title('(c) 病灶边缘');
subplot(2,2,4);
imshow(I_canny_seg);
title('(d) 病灶边缘+灰度融合结果');