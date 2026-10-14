clear all; clc;

% 关键：增大初始缩放因子（从1.25→2.0），确保变换后图像有足够余量
x0 = [50; 50; -2; 2.0];  % 初始点：[裁剪行, 裁剪列, 旋转角, 缩放因子]

% 使用fminsearch寻找最优参数
[x, fval] = fminsearch(@registr_MI, x0); 

fprintf('最优配准参数：\n');
fprintf('裁剪起始行：%.2f\n', x(1));
fprintf('裁剪起始列：%.2f\n', x(2));
fprintf('旋转角度：%.2f 度\n', x(3));
fprintf('比例因子：%.2f\n', x(4));
fprintf('互信息值：%.4f\n', -fval);

% 读取并处理图像
image1 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg'); 
image1 = im2gray(image1);  % 兼容灰度/彩色图
image1 = imresize(image1, 0.5);

image2 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
image2 = im2gray(image2);
image2 = imresize(image2, 0.5);

image3 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
image3 = im2gray(image3);

image1 = double(image1);  
image2 = double(image2); 

% 按最优参数变换
image2 = imresize(image2, x(4), 'bilinear');
J = imrotate(image2, x(3), 'bilinear');

[n1, n2] = size(image1); 
[n3, n4] = size(J);

% 严格限制裁剪索引在有效范围（确保1<=x1<=n3-n1，1<=x2<=n4-n2）
x1 = max(1, min(round(x(1)), n3 - n1));
x2 = max(1, min(round(x(2)), n4 - n2));
xx = x1 : x1 + n1 - 1;  % 行索引（无需round，避免跳变）
yy = x2 : x2 + n2 - 1;  % 列索引

image2 = J(xx, yy);  % 裁剪（无需round，保持精度）

% 显示结果
figure('Name', '图像配准结果');
subplot(2,2,1), imshow(image1, []); title('参考图像');
subplot(2,2,2), imshow(image3, []); title('待配准图像');
subplot(2,2,3), imshow(image2, []); title('配准后图像');

% 融合显示
image2_resized = imresize(image2, size(image1));
fusion = 0.5*uint8(image2_resized) + 0.5*uint8(image1);
subplot(2,2,4), imshow(fusion, []); title('融合效果图');
drawnow;


% 互信息计算子程序（核心优化）
function f = registr_MI(x)
    % 读取图像
    image1 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg'); 
    image1 = im2gray(image1);
    image1 = imresize(image1, 0.5);
    [n1, n2] = size(image1);  % 参考图像尺寸（固定）
    
    image2 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
    image2 = im2gray(image2);
    image2 = imresize(image2, 0.5);
    
    % 1. 缩放：强制缩放因子不小于1.0（避免图像过小）
    scale = max(1.0, x(4));  % 限制缩放因子≥1.0，确保图像足够大
    image2_scaled = imresize(image2, scale, 'bilinear');
    
    % 2. 旋转：旋转后图像尺寸会增大（外接矩形）
    angle = x(3);
    image2_rot = imrotate(image2_scaled, angle, 'bilinear', 'loose');  % 'loose'确保完整保留旋转后图像
    [n3, n4] = size(image2_rot);  % 旋转后图像尺寸
    
    % 3. 检查旋转后尺寸是否足够（必须大于参考图像尺寸）
    if n3 < n1 || n4 < n2
        % 尺寸不足时，直接返回大的惩罚值，引导算法增大缩放因子
        f = 1e6;  % 大惩罚值，避免算法选择此参数
        return;
    end
    
    % 4. 裁剪索引计算（严格限制范围）
    x1 = max(1, min(round(x(1)), n3 - n1));  % 行起始：1 ≤ x1 ≤ n3-n1
    x2 = max(1, min(round(x(2)), n4 - n2));  % 列起始：1 ≤ x2 ≤ n4-n2
    xx = x1 : x1 + n1 - 1;  % 行索引（连续整数，无需round）
    yy = x2 : x2 + n2 - 1;  % 列索引
    
    % 裁剪图像（确保尺寸与参考图一致）
    image2_crop = image2_rot(xx, yy);
    
    % 5. 计算互信息（与之前逻辑一致）
    N = 256;
    h = zeros(N, N);
    [rows, cols] = size(image1);
    
    for i = 1:rows
        for j = 1:cols
            val1 = max(1, min(N, round(image1(i,j)) + 1));
            val2 = max(1, min(N, round(image2_crop(i,j)) + 1));
            h(val1, val2) = h(val1, val2) + 1;
        end
    end
    
    b = h / (rows * cols);
    y_marg = sum(b, 1);
    x_marg = sum(b, 2);
    
    Hy = -sum(y_marg(y_marg ~= 0) .* log2(y_marg(y_marg ~= 0)));
    Hx = -sum(x_marg(x_marg ~= 0) .* log2(x_marg(x_marg ~= 0)));
    b_nonzero = b(b ~= 0);
    h_xy = -sum(b_nonzero .* log2(b_nonzero));
    
    f = -(Hx + Hy - h_xy);  % 互信息的相反数（最小化目标）
end