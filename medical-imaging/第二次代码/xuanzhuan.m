% 1. 读取DICOM图像与元数据
dcmPath = "E:\医学图像处理\series-000006\image-000012.dcm";
I = dicomread(dcmPath);
info = dicominfo(dcmPath);

% 2. 窗宽窗位处理（确保医学影像正确显示）
if isfield(info, 'WindowCenter') && isfield(info, 'WindowWidth') && ...
    ~isempty(info.WindowCenter) && ~isempty(info.WindowWidth)
    wc = info.WindowCenter(1);  % 窗位
    ww = info.WindowWidth(1);   % 窗宽
else
    wc = mean(double(I(:)));    % 无窗宽窗位时用均值
    ww = range(double(I(:)));   % 用灰度范围作为窗宽
    if ww == 0, ww = 1; end     % 避免窗宽为0
end

% 3. 图像预处理（归一化+窗宽窗位映射）
I_double = double(I);
I_min = min(I_double(:));
I_max = max(I_double(:));

if I_max == I_min
    A = zeros(size(I_double));  % 单灰度图像特殊处理
else
    A = (I_double - I_min) / (I_max - I_min);  % 归一化到0-1
    
    % 计算窗宽窗位对应的归一化范围
    low_in = (wc - ww/2 - I_min) / (I_max - I_min);
    high_in = (wc + ww/2 - I_min) / (I_max - I_min);
    low_in = max(0, min(low_in, 1));
    high_in = max(0, min(high_in, 1));
    if low_in >= high_in, low_in=0; high_in=1; end  % 确保参数合法
    
    A = imadjust(A, [low_in, high_in], [0, 1]);  % 应用窗宽窗位
end

% 4. 旋转操作（包含顺时针和逆时针）
% 顺时针旋转（角度为负数）
rot_cw40 = imrotate(A, -40, 'bilinear');  % 顺时针40°，不裁切

% 逆时针旋转（角度为正数）
rot_ccw30 = imrotate(A, 30, 'bilinear');   % 逆时针30°，不裁切
rot_ccw90 = imrotate(A, 90, 'bilinear');   % 逆时针90°，不裁切

% 5. 显示结果（对比顺时针与逆时针）
figure('Name', '医学影像旋转对比');
subplot(2,3,1); imshow(A); title('1. 原影像');

% 顺时针旋转
subplot(2,3,2); imshow(rot_cw40); title('2. 顺时针40°（不裁切）');

% 逆时针旋转
subplot(2,3,3); imshow(rot_ccw30); title('3. 逆时针30°（不裁切）');
subplot(2,3,4); imshow(rot_ccw90); title('4. 逆时针90°（不裁切）');