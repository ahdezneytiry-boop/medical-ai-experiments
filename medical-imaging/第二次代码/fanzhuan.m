% 读取DICOM文件
dcm_path ="E:\医学图像处理\series-000006\image-000022.dcm";  % DICOM路径
I = dicomread(dcm_path);
info = dicominfo(dcm_path);

% 获取窗宽窗位
wc = info.WindowCenter(1);
ww = info.WindowWidth(1);
display_range = [wc - ww/2, wc + ww/2];

% 镜像操作（基于flip函数）
mirror_vertical = flipud(I);  % 垂直镜像（上下翻转）
mirror_horizontal = fliplr(I);  % 水平镜像（左右翻转）

% 显示结果
figure;
subplot(2,2,1); imshow(I, display_range); title('原影像');
subplot(2,2,2); imshow(mirror_vertical, display_range); title('垂直镜像');
subplot(2,2,3); imshow(mirror_horizontal, display_range); title('水平镜像');