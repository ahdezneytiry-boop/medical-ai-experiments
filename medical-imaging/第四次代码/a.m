% 三种边缘检测算子（Canny/Sobel/Roberts）完整对比 - 病灶分割代码
clear; clc; close all;

% 1. 读取并预处理图像（三种算子共用同一预处理，保证对比公平）
pngPath = 'C:\Users\ahdezneytiry-boop1\Desktop\018.png';
I1 = imread(pngPath);
I1_gray = rgb2gray(I1);
I1_gray = im2double(I1_gray);

% 高斯滤波去噪（σ=0.6，轻度去噪保边缘）+ 对比度增强（适配高对比度病灶）
I_blur = imgaussfilt(I1_gray, 0.6);
I_enhance = adapthisteq(I_blur, 'ClipLimit', 0.03);

% 创建大窗口（3行3列子图，清晰展示每种算子的边缘+分割+融合结果）
figure('Position', [100, 100, 1500, 900]);
sgtitle('三种边缘检测算子的病灶分割效果完整对比', 'FontSize', 16, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% 2. Canny算子（抗噪最优、边缘最纯净）
% -------------------------------------------------------------------------
I_canny = edge(I_enhance, 'canny', [0.15, 0.35]);  % 双阈值筛选强/弱边缘
I_canny_fill = imclose(I_canny, strel('disk', 2));  % 闭合微小缺口
I_canny_seg = zeros(size(I1_gray));  % 纯二值分割结果
I_canny_seg(I_canny_fill) = 1;
I_canny_fuse = I1_gray;  % 灰度融合结果
I_canny_fuse(I_canny_fill) = 1;

% Canny子图（第1行）
subplot(3,3,1); imshow(I_canny); title('(1) Canny-病灶边缘', 'FontSize', 12);
subplot(3,3,2); imshow(I_canny_seg); title('(2) Canny-二值分割', 'FontSize', 12);
subplot(3,3,3); imshow(I_canny_fuse); title('(3) Canny-灰度融合', 'FontSize', 12);

% -------------------------------------------------------------------------
% 3. Sobel算子（抗噪中等、边缘较粗，速度快）- 重点展示，无遗漏
% -------------------------------------------------------------------------
I_sobel = edge(I_enhance, 'sobel', 0.2);  % 单阈值0.2，平衡边缘数量与噪声
I_sobel_fill = imclose(I_sobel, strel('disk', 3));  % 增大结构半径，填补边缘缺口
I_sobel_seg = zeros(size(I1_gray));  % 纯二值分割结果
I_sobel_seg(I_sobel_fill) = 1;
I_sobel_fuse = I1_gray;  % 灰度融合结果
I_sobel_fuse(I_sobel_fill) = 1;

% Sobel子图（第2行）
subplot(3,3,4); imshow(I_sobel); title('(4) Sobel-病灶边缘', 'FontSize', 12);
subplot(3,3,5); imshow(I_sobel_seg); title('(5) Sobel-二值分割', 'FontSize', 12);
subplot(3,3,6); imshow(I_sobel_fuse); title('(6) Sobel-灰度融合', 'FontSize', 12);

% -------------------------------------------------------------------------
% 4. Roberts算子（抗噪最差、边缘细碎，速度最快）
% -------------------------------------------------------------------------
I_roberts = edge(I_enhance, 'roberts', 0.18);  % 阈值略小，避免边缘丢失
I_roberts_clean = imerode(I_roberts, strel('disk', 1));  % 腐蚀去除细碎噪声
I_roberts_fill = imclose(I_roberts_clean, strel('disk', 3));  % 闭合修复边缘
I_roberts_seg = zeros(size(I1_gray));  % 纯二值分割结果
I_roberts_seg(I_roberts_fill) = 1;
I_roberts_fuse = I1_gray;  % 灰度融合结果
I_roberts_fuse(I_roberts_fill) = 1;

% Roberts子图（第3行）
subplot(3,3,7); imshow(I_roberts); title('(7) Roberts-病灶边缘', 'FontSize', 12);
subplot(3,3,8); imshow(I_roberts_seg); title('(8) Roberts-二值分割', 'FontSize', 12);
subplot(3,3,9); imshow(I_roberts_fuse); title('(9) Roberts-灰度融合', 'FontSize', 12);