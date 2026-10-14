% 1. 读取DICOM影像与元数据
dcm_path = "E:\医学图像处理\series-000006\image-000010.dcm";
I = dicomread(dcm_path);
info = dicominfo(dcm_path);
% 2. 安全获取窗宽窗位
if isfield(info, 'WindowCenter') && isfield(info, 'WindowWidth') && ...
        ~isempty(info.WindowCenter) && ~isempty(info.WindowWidth)
    wc = double(info.WindowCenter(1));  % 转为double避免整数溢出
    ww = double(info.WindowWidth(1));
    % 确保窗宽为正数（防止无效值）
    if ww <= 0
        warning('窗宽无效，使用默认值');
        ww = 1000;  % 给一个合理的默认窗宽
    end
else
    % 无窗宽窗位时，用影像自身的灰度范围
    wc = mean(double(I(:)));  % 均值作为窗位
    ww = range(double(I(:))); % 灰度范围作为窗宽
    if ww == 0  % 图像为单灰度时
        ww = 1;  % 避免窗宽为0
    end
end
% 3. 归一化原始图像（处理极端情况：如所有像素值相同）
I_double = double(I);  % 转为double类型
I_min = min(I_double(:));
I_max = max(I_double(:));
% 处理I_max == I_min（图像无灰度变化）
if I_max == I_min
    I_norm = zeros(size(I_double));  % 单灰度图像归一化为0（或1，不影响显示）
else
    I_norm = (I_double - I_min) / (I_max - I_min);  % 正常归一化到0-1
end
% 4. 计算窗宽窗位对应的归一化范围（强制满足low_in < high_in且在[0,1]内）
low_raw = wc - ww/2;    % 原始窗宽下限
high_raw = wc + ww/2;   % 原始窗宽上限
% 转换为归一化范围
if I_max == I_min
    % 单灰度图像，直接用0-1
    low_in = 0;
    high_in = 1;
else
    low_in = (low_raw - I_min) / (I_max - I_min);
    high_in = (high_raw - I_min) / (I_max - I_min);
end
% 强制边界：确保在[0,1]内
low_in = max(0, min(low_in, 1));
high_in = max(0, min(high_in, 1));

% 强制low_in < high_in（若相等，手动拉开微小差距）
if low_in >= high_in
    low_in = 0;
    high_in = 1;  % 用全范围避免错误
end
% 5. 应用窗宽窗位
I_vis = imadjust(I_norm, [low_in, high_in], [0, 1]);
% 6. 执行缩放操作
scale_2x = imresize(I_vis, 2, 'bilinear');       % 放大2倍
scale_05x = imresize(I_vis, 0.5, 'bilinear');    % 缩小0.5倍
scale_fixed = imresize(I_vis, [200, 200], 'nearest');  % 固定尺寸
% 7. 显示结果
figure;
subplot(2,2,1); imshow(I_vis, [0,1]); title('原影像');
subplot(2,2,2); imshow(scale_2x, [0,1]); title('放大2倍');
subplot(2,2,3); imshow(scale_05x, [0,1]); title('缩小0.5倍');
subplot(2,2,4); imshow(scale_fixed, [0,1]); title('固定200x200');