% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 转换为 uint8 类型
I = im2uint8(I);

% 生成各类噪声图像
% a. 原始图像（已读取）

% b. 高斯噪声（方差0.01）
% 参数：均值=0，方差=0.01
I_gauss001 = imnoise(I, 'gaussian', 0, 0.01);

% c. 椒盐噪声（默认密度0.05）
% 特点：随机出现白色（盐）和黑色（椒）噪点，密度0.05表示5%像素被污染
I_salt = imnoise(I, 'salt & pepper');

% d. 泊松噪声（基于图像亮度的自然噪声）
% 特点：噪声强度与像素灰度值相关（亮区噪声更明显），模拟成像传感器噪声
I_poisson = imnoise(I, 'poisson');

% e. 乘性噪声（斑点噪声，默认方差0.05）
% 特点：噪声与像素值成比例，常见于超声、SAR图像
I_speckle = imnoise(I, 'speckle', 0.05);  % 0.05为噪声方差，值越大噪声越强

% f. 高斯噪声（方差0.001）
% 参数：均值=0，方差=0.001（噪声较弱，细节保留更好）
I_gauss0001 = imnoise(I, 'gaussian', 0, 0.001);

% 6图整合显示（2行3列布局）
figure('Name', '各类噪声图像对比', 'Position', [100 100 1200 800]);  % 调整窗口大小，避免拥挤

% a. 原始图像
subplot(2,3,1);
imshow(I);
title('a. 原始图像');

% b. 高斯噪声（方差0.01）
subplot(2,3,2);
imshow(I_gauss001);
title('b. 高斯噪声（方差0.01）');

% c. 椒盐噪声
subplot(2,3,3);
imshow(I_salt);
title('c. 椒盐噪声（密度0.05）');

% d. 泊松噪声
subplot(2,3,4);
imshow(I_poisson);
title('d. 泊松分布噪声');

% e. 乘性噪声（斑点噪声）
subplot(2,3,5);
imshow(I_speckle);
title('e. 乘性噪声（方差0.05）');

% f. 高斯噪声（方差0.001）
subplot(2,3,6);
imshow(I_gauss0001);
title('f. 高斯噪声（方差0.001）');