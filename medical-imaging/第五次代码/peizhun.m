clear all;
clc;

% 初始优化参数：[行偏移, 列偏移, 旋转角度, 缩放因子]
x0 = [50; 50; -2; 1.25];  
% 调用fminsearch优化（最小化互信息的相反数）
[x, fval] = fminsearch(@registr_MI, x0);

% 读取并预处理参考图像（图片1）和待配准图像（图片2）
image1 = imread("C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg");   % 参考图像（如MRI）
image1 = imresize(image1, 0.5); % 缩小2倍
image2 = imread("C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg");    % 待配准图像（如CT）
image2 = imresize(image2, 0.5); % 缩小2倍
image3 = imread("C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg");    % 原始待配准图像（用于显示）
image3 = imresize(image3, 0.5); % 同步缩小

% 格式转换与几何变换（缩放→旋转→偏移）
image1 = double(image1);
image2 = double(image2);
image2 = imresize(image2, x(4), 'bilinear'); % 按最优比例缩放
J = imrotate(double(image2), x(3), 'bilinear'); % 按最优角度旋转

% 提取匹配区域
[n1, n2] = size(image1);
xx = round(1:n1 + x(1)); % 行偏移
yy = round(1:n1 + x(2)); % 列偏移
image2 = round(J(xx, yy)); % 配准后图像

% 显示结果
subplot(2, 2, 1), imshow(uint8(image1)), title('参考图像 Image 1');
subplot(2, 2, 2), imshow(image3), title('待配准图像 Image 2');
subplot(2, 2, 3), imshow(uint8(image2)), title('配准后的图像 Image 2');
subplot(2, 2, 4), imshow(0.5*uint8(image2) + 0.5*uint8(image1)), title('简单融合后的效果图');

% 输出最优参数
disp('最优配准参数 x：'); disp(x);
disp('互信息的相反数 fval：'); disp(fval);