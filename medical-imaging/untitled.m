clear all; clc;
I1 = dicomread("E:\背景图像.dcm"); 
I2 = dicomread("E:\增强图像.dcm"); 
% 预处理
if size(I1) ~= size(I2)
    error('两幅图像尺寸不一致，无法减法运算！'); 
end
I1 = double(I1);
I2 = double(I2);
J = I2 - I1;  % 增强图-背景图
J(J < 0) = 0;  
J_norm = mat2gray(J);
% 显示对比结果
figure('Position', [200, 200, 1200, 400]);
subplot(1,3,1); imshow(I1, []); title('背景图像（无对比剂）', 'FontSize', 12);
subplot(1,3,2); imshow(I2, []); title('增强图像（含对比剂）', 'FontSize', 12);
subplot(1,3,3); imshow(J_norm); title('减法运算结果（突出血管）', 'FontSize', 12);