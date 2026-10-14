% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 均值滤波（5×5邻域）
H_avg = fspecial('average', 5);
K_smooth = filter2(H_avg, I);  
% 保持与原始图像相同的数据类型
K_smooth = cast(K_smooth, class(I));

% 同图显示两张图像
figure('Name', '原始图像与均值滤波对比');

% 原始图像
subplot(1,2,1);
imshow(I, []);  % 自动适配灰度范围
title('原始医学影像');

% 均值滤波后图像
subplot(1,2,2);
imshow(K_smooth, []);  % 同样自动适配，避免模糊
title('平滑操作（5×5均值滤波）后图像');