% 读取DICOM文件
dcm_path ="E:\医学图像处理\series-000006\image-000021.dcm";  % DICOM路径
I = dicomread(dcm_path);
info = dicominfo(dcm_path);

% 获取窗宽窗位
wc = info.WindowCenter(1);
ww = info.WindowWidth(1);
display_range = [wc - ww/2, wc + ww/2];

% 定义平移量
dx = 80;   % 水平平移量
dy = 80;   % 垂直平移量

% 获取原图尺寸
[rows, cols] = size(I);

% 计算扩展后图像尺寸（避免裁剪）
new_rows = rows + abs(dy);  % 垂直方向扩展
new_cols = cols + abs(dx);  % 水平方向扩展

% 初始化扩展矩阵（保留原始数据类型）
trans_expand = zeros(new_rows, new_cols, class(I));

% 计算原图在新矩阵中的起始位置
start_i = max(1, 1 + dy);  % dy为正（下移）时，起始行后移
start_j = max(1, 1 + dx);  % dx为正（右移）时，起始列后移

% 填充原图像素到扩展矩阵
trans_expand(start_i:start_i+rows-1, start_j:start_j+cols-1) = I;

% 显示结果
figure;
subplot(1,2,1); imshow(I, display_range); title('原影像');
subplot(1,2,2); imshow(trans_expand, display_range); title('扩展平移后');