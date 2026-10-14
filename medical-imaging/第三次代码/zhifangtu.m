% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 查看原始图像的灰度范围
I_min = min(I(:));  % 最小灰度值
I_max = max(I(:));  % 最大灰度值
fprintf('原始图像灰度范围：[%d, %d]\n', I_min, I_max);

% 示例：取原始灰度的10%~90%作为输入范围，映射到全范围[0, max(I)]
p = prctile(I(:), [10, 90]);  % 取10%和90%分位数作为有效范围
G = imadjust(I, [p(1)/I_max, p(2)/I_max], [0, 1]);  % 不反转，正向映射

% 同图显示原始图像和增强图像
figure;
subplot(1,2,1);  % 1行2列第1个位置
imshow(I, []);  % 用[]自动调整显示范围，避免原始图像过暗/过亮
title('原始医学影像');

subplot(1,2,2);  % 1行2列第2个位置
imshow(G, []);  % 同样自动调整显示范围
title('对比度增强后图像');