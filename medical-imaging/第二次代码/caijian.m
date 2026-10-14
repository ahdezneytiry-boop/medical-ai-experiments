% 1. 读取DICOM图像与元数据
dcmPath = "E:\医学图像处理\series-000006\image-000016.dcm";
I = dicomread(dcmPath);
info = dicominfo(dcmPath);

% 2. 窗宽窗位处理（确保裁剪前后灰度显示一致）
if isfield(info, 'WindowCenter') && isfield(info, 'WindowWidth') && ...
    ~isempty(info.WindowCenter) && ~isempty(info.WindowWidth)
    wc = info.WindowCenter(1);  % 窗位
    ww = info.WindowWidth(1);   % 窗宽
else
    wc = mean(double(I(:)));    % 无窗宽窗位时用均值
    ww = range(double(I(:)));   % 用灰度范围作为窗宽
    if ww == 0, ww = 1; end     % 避免窗宽为0
end

% 3. 图像预处理（归一化+窗宽窗位映射，生成可视化图像）
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

% 4. 剪切（裁剪）操作
% 方式1：交互式裁剪
figure('Name', '请框选裁剪区域（拖动鼠标，双击确认）');
imshow(A); title('原影像（请框选裁剪区域）');
cropped_interactive = imcrop(A);  % 交互式裁剪

% 5. 显示结果
figure('Name', '医学影像剪切结果');
subplot(2,2,1); imshow(A); title('1. 原影像');
subplot(2,2,2); imshow(cropped_interactive); title('2. 交互式裁剪结果');
subplot(2,2,3); imshow(cropped_fixed); title('3. 指定区域裁剪结果');