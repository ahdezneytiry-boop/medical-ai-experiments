dcm_path ="E:\医学图像处理\series-000006\image-000020.dcm";  % DICOM路径
I = dicomread(dcm_path);
info = dicominfo(dcm_path);

% 获取窗宽窗位
wc = info.WindowCenter(1);
ww = info.WindowWidth(1);
display_range = [wc - ww/2, wc + ww/2];

% 定义平移量
dx = 50;   % 正数右移，负数左移
dy = 30;   % 正数下移，负数上移

% 获取原图尺寸
[rows, cols] = size(I);

% 初始化输出图像
trans_noexpand = zeros(rows, cols, class(I));

% 像素级平移
for i = 1:rows
    for j = 1:cols
        % 计算原图像对应坐标
        orig_i = i - dy;  % 垂直方向：新行 = 原行 + dy → 原行 = 新行 - dy
        orig_j = j - dx;  % 水平方向：新列 = 原列 + dx → 原列 = 新列 - dx
        
        % 仅保留原图范围内的像素
        if orig_i >= 1 && orig_i <= rows && orig_j >= 1 && orig_j <= cols
            trans_noexpand(i, j) = I(orig_i, orig_j);
        end
    end
end

% 显示结果
figure;
subplot(1,2,1); imshow(I, display_range); title('原影像');
subplot(1,2,2); imshow(trans_noexpand, display_range); title('不扩展平移后');