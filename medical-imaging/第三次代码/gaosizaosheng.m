% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 添加高斯噪声
% 'gaussian'表示添加高斯噪声，后两个参数分别为：均值=0，方差=0.01
% 方差越大，噪声越强,图像越模糊；
I_gauss = imnoise(I, 'gaussian', 0, 0.01);

% 中值滤波（3×3邻域）
% 中值滤波对高斯噪声的处理效果较弱
% [3 3]窗口：取3×3邻域内像素的中值，窗口过大会导致图像模糊
K_med_gauss = medfilt2(I_gauss, [3 3]);

% 均值滤波（5×5邻域）
% 均值滤波通过邻域平均平滑高斯噪声，核大小决定平滑强度
% fspecial('average', 5)创建5×5均值核（每个元素=1/25），核越大平滑越强但边缘越模糊
H_avg = fspecial('average', 5);
K_avg_gauss = filter2(H_avg, I_gauss);  % 去除/255，避免16位DICOM图像灰度压缩
K_avg_gauss = cast(K_avg_gauss, class(I));  % 保持数据类型一致

% 自适应维纳滤波（3×3邻域）
% 维纳滤波是处理高斯噪声的最优方法之一，能根据局部灰度方差自适应调整滤波强度
% [3 3]窗口：在3×3邻域内计算局部均值和方差，平衡去噪与保边
K_wiener = wiener2(I_gauss, [3 3]);

% 四图对比显示
figure('Name', '高斯噪声及滤波处理对比');

% 1. 原始图像
subplot(2,2,1);
imshow(I, []);
title('原始医学影像');

% 2. 添加高斯噪声后图像
subplot(2,2,2);
imshow(I_gauss, []);
title('添加高斯噪声（方差0.01）后图像');

% 3. 均值滤波后图像（高斯噪声更适用）
subplot(2,2,3);
imshow(K_avg_gauss, []);
title('均值滤波（5×5邻域）后图像');

% 4. 自适应维纳滤波后图像（最优效果）
subplot(2,2,4);
imshow(K_wiener, []);
title('自适应维纳滤波（3×3邻域）后图像');