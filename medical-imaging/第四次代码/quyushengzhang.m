clear; clc; close all;

% 1. 图像读取与预处理
imgPath = 'C:\Users\ahdezneytiry-boop1\Desktop\018.png';
I = imread(imgPath);
I_gray = rgb2gray(I); 
% 针对性增强左侧病灶区域的对比度
I_gray = imadjust(I_gray, [0.1, 0.9], [0, 1]);
I_double = double(I_gray);
subplot(1,2,1);
imshow(I_gray);
title('(a) 精准预处理后图像及种子点');

% 2. 定位左侧病灶核心种子点
seedx = [440, 470, 500];   % 左侧高亮度病灶最亮区域横坐标
seedy = [270, 300, 330];   % 左侧高亮度病灶最亮区域纵坐标
hold on;
plot(seedx, seedy, 'gs', 'LineWidth', 2);

% 3. 构建种子点标记图像
markerim = I_double == I_double(seedy(1), seedx(1));
for i = 2:length(seedx)
    markerim = markerim | (I_double == I_double(seedy(i), seedx(i)));
end

% 4. 动态调整灰度阈值
thresh = [15, 15, 15];  % 适度增大阈值，确保病灶完整覆盖
maskim = zeros(size(I_double));
for i = 1:length(seedx)
    g = abs(I_double - I_double(seedy(i), seedx(i))) <= thresh(i);
    maskim = maskim | g;
end

% 5. 图像重建与分割结果显示
[g, ~] = bwlabel(imreconstruct(markerim, maskim), 8);
g = mat2gray(g);
subplot(1,2,2);
imshow(g);
title('(b) 最终精准区域生长分割结果');