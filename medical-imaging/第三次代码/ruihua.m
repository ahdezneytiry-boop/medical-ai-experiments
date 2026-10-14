% 读取DICOM图像
I = dicomread('E:\医学图像处理\series-000006\image-000006.dcm');

% 调整锐化强度（α=0.8，适当增强锐化效果），并避免除以255导致灰度压缩
H_unsharp = fspecial('unsharp', 0.8);  % α值越大，锐化越强
K_sharp = filter2(H_unsharp, I);       % 保留原始灰度范围
K_sharp = cast(K_sharp, class(I));     % 保持与原图相同的数据类型

% 同图显示（1行2列对比）
figure('Name', '原始图像与锐化对比');

% 原始图像
subplot(1,2,1);
imshow(I, []);  % 自动适配灰度显示
title('原始医学影像');

% 锐化后图像
subplot(1,2,2);
imshow(K_sharp, []);  % 同样自动适配，避免亮度异常
title('锐化操作（非锐化掩模）后图像');