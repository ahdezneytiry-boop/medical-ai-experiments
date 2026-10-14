clear all; clc;

% 定义初始点x0，包含裁剪起始行、裁剪起始列、旋转角度、比例因子
x0 = [50; 50; -2; 1.25];  

% 使用fminsearch函数寻找最优变换参数
[x, fval] = fminsearch(@registr_MI, x0); 

fprintf('最优配准参数：\n');
fprintf('裁剪起始行：%.2f\n', x(1));
fprintf('裁剪起始列：%.2f\n', x(2));
fprintf('旋转角度：%.2f 度\n', x(3));
fprintf('比例因子：%.2f\n', x(4));
fprintf('互信息值：%.4f\n', -fval);

% 读取参考图像（转为灰度图，兼容彩色/灰度输入）
image1 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg'); 
image1 = im2gray(image1);  % 用im2gray替代rgb2gray，兼容灰度图
image1 = imresize(image1, 0.5);  % 缩小2倍

% 读取待配准图像（转为灰度图）
image2 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
image2 = im2gray(image2);  % 兼容灰度图
image2 = imresize(image2, 0.5);  % 缩小2倍

% 读取原始待配准图像用于显示
image3 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
image3 = im2gray(image3);  % 兼容灰度图

% 转换为double类型
image1 = double(image1);  
image2 = double(image2); 

% 按最优参数处理待配准图像
image2 = imresize(image2, x(4), 'bilinear');  % 缩放
J = imrotate(image2, x(3), 'bilinear');  % 旋转

% 获取参考图像尺寸
[n1, n2] = size(image1); 
% 获取旋转后图像尺寸
[n3, n4] = size(J);

% 修正裁剪索引（避免越界）
x1 = max(1, min(x(1), n3 - n1));  % 行起始索引限制在有效范围
x2 = max(1, min(x(2), n4 - n2));  % 列起始索引限制在有效范围
xx = round(x1 : x1 + n1 - 1);  % 行索引（确保长度为n1）
yy = round(x2 : x2 + n2 - 1);  % 列索引（确保长度为n2）

% 裁剪得到配准后图像（确保尺寸匹配）
image2 = round(J(xx, yy)); 

% 显示结果（强制刷新图像窗口）
figure('Name', '图像配准结果');  % 新建窗口
subplot(2, 2, 1), imshow(image1, []); title('参考图像 Image 1');
subplot(2, 2, 2), imshow(image3, []); title('待配准图像 Image 2');
subplot(2, 2, 3), imshow(image2, []); title('配准后的图像 Image 2');

% 融合显示（统一尺寸）
image2_resized = imresize(image2, size(image1));
fusion = 0.5 * uint8(image2_resized) + 0.5 * uint8(image1);
subplot(2, 2, 4), imshow(fusion, []); title('简单融合后的效果图');
drawnow;  % 强制刷新显示


% 互信息计算子程序
function f = registr_MI(x)
    % 读取图像（转为灰度图，兼容彩色/灰度输入）
    image1 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片2.jpg'); 
    image1 = im2gray(image1);  % 替换为im2gray
    image1 = imresize(image1, 0.5);
    
    image2 = imread('C:\Users\ahdezneytiry-boop1\Downloads\图片1.jpg'); 
    image2 = im2gray(image2);  % 替换为im2gray
    image2 = imresize(image2, 0.5);
    
    % 转换为double类型
    image1 = double(image1); 
    image2 = double(image2);
    
    % 几何变换
    image2 = imresize(image2, x(4), 'bilinear');  % 缩放
    J = imrotate(image2, x(3), 'bilinear');  % 旋转
    
    % 归一化处理
    J = abs(J) * 255 / max(J(:));
    
    % 获取图像尺寸
    [n1, n2] = size(image1);
    [n3, n4] = size(J);
    
    % 检查缩放因子是否过小
    if n1 > n3 - x(1)/2
        f = 1000;
        message = strvcat('缩放因子过小', '按Ctrl+C停止', '增大x0(4)后重启');
        disp('按Ctrl+C停止');
        errordlg(message);
        pause;
    else
        % 处理裁剪索引超出范围的情况（限制在有效范围）
        x1 = max(1, min(x(1), n3 - n1));
        x2 = max(1, min(x(2), n4 - n2));
        if x(1) ~= x1 || x(2) ~= x2
            warning('裁剪索引超出范围，已自动修正');
        end
    end
    
    % 计算裁剪索引（确保长度匹配）
    xx = round(x1 : x1 + n1 - 1);
    yy = round(x2 : x2 + n2 - 1);
    
    % 裁剪图像以匹配尺寸
    image2_crop = round(J(xx, yy));
    
    % 计算联合直方图
    N = 256;  % 灰度级
    h = zeros(N, N);
    [rows, cols] = size(image1);
    
    for i = 1:rows
        for j = 1:cols
            % 确保灰度值在有效范围内（0-255）
            val1 = max(1, min(N, round(image1(i,j)) + 1));
            val2 = max(1, min(N, round(image2_crop(i,j)) + 1));
            h(val1, val2) = h(val1, val2) + 1;
        end
    end
    
    % 计算概率分布
    b = h / (rows * cols);  % 联合概率
    y_marg = sum(b, 1);     % 图像1边际概率（列求和）
    x_marg = sum(b, 2);     % 图像2边际概率（行求和）
    
    % 计算边际熵（避免log2(0)错误）
    Hy = -sum(y_marg(y_marg ~= 0) .* log2(y_marg(y_marg ~= 0)));
    Hx = -sum(x_marg(x_marg ~= 0) .* log2(x_marg(x_marg ~= 0)));
    
    % 计算联合熵
    b_nonzero = b(b ~= 0);
    h_xy = -sum(b_nonzero .* log2(b_nonzero));
    
    % 互信息的相反数（用于最小化）
    f = -(Hx + Hy - h_xy);
end