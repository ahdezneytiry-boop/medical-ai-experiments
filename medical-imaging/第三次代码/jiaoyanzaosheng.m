% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 添加椒盐噪声（参数说明：默认噪声密度为0.05，即5%的像素被污染）
% 'salt & pepper'表示添加椒盐噪声：白色像素（盐噪声）和黑色像素（椒噪声）随机分布
I_salt = imnoise(I, 'salt & pepper');  % 可通过添加参数调整密度

% 中值滤波
% 参数[3 3]表示使用3×3的邻域窗口：对每个像素，取其3×3邻域内的像素值的中值作为该像素的新值
% 能有效去除椒盐噪声，同时保留图像边缘细节
K_med = medfilt2(I_salt, [3 3]);  

% 均值滤波
% fspecial('average', 5)创建5×5的均值滤波核：每个元素值为1/(5×5)=1/25，实现邻域像素的平均
% 均值滤波劣势：虽能平滑噪声，但会模糊图像边缘
H_avg = fspecial('average', 5);
K_avg = filter2(H_avg, I_salt);  % 去除/255，避免灰度范围压缩
K_avg = cast(K_avg, class(I));   % 保持与原图一致的数据类型，确保显示正常

% 四图对比显示
figure('Name', '椒盐噪声及滤波处理对比');

% 1. 原始图像
subplot(2,2,1);
imshow(I, []);
title('原始医学影像');

% 2. 添加椒盐噪声后图像
subplot(2,2,2);
imshow(I_salt, []);
title('添加椒盐噪声后图像');

% 3. 中值滤波后图像
subplot(2,2,3);
imshow(K_med, []);
title('中值滤波（3×3邻域）后图像');

% 4. 均值滤波后图像
subplot(2,2,4);
imshow(K_avg, []);
title('均值滤波（5×5邻域）后图像');