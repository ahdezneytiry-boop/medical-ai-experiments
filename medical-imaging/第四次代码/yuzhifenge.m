% 迭代阈值法
clear; clc; close all;

% 1. 读取并预处理图像
pngPath = 'C:\Users\ahdezneytiry-boop1\Desktop\018.png';
I1 = imread(pngPath);
I1_gray = rgb2gray(I1);
I1_gray = im2double(I1_gray);

% 增强病灶与正常组织的对比度
I1_gray = adapthisteq(I1_gray, 'ClipLimit', 0.05); 

subplot(1,2,1);
imshow(I1_gray);
title('(a) 对比度增强');

% 2. 迭代阈值优化
I = I1_gray;
d = false;
iter_cnt = 0;
% 初始阈值设为图像前85%灰度
T = prctile(I(:), 85);  
max_iter = 30;
thresh_diff = 0.001;

while ~d && iter_cnt < max_iter
    u1 = I <= T;  % 正常组织
    u2 = I > T;   % 病灶（高灰度）
    mean1 = mean(I(u1), 'omitnan');
    mean2 = mean(I(u2), 'omitnan');
    Tn = (mean1 + mean2) / 2;
    d = abs(Tn - T) < thresh_diff;
    T = Tn;
    iter_cnt = iter_cnt + 1;
end

% 3. 形态学后处理
I_seg = zeros(size(I));
I_seg(u2) = 1;
% 去除小面积噪声
I_seg = bwareaopen(I_seg, 50);  
% 闭合病灶内小空洞
I_seg = imclose(I_seg, strel('disk', 2));

subplot(1,2,2);
imshow(I_seg);
title('(b)分割结果');
fprintf('病灶分割阈值：%.4f\n', T);