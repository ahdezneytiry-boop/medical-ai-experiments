%% 像素灰度极值法融合
% 1. 读入图像
ref_img = imread('E:\\ML_Results_Validity\\017.png');   % 参考图像（MRI）
moving_img = imread('E:\\ML_Results_Validity\\017 (1).png'); % 待配准CT图像

% 2. 转灰度图
ref_img = rgb2gray(ref_img);
moving_img = rgb2gray(moving_img);

% 3. 数据类型转换
ref_double = im2double(ref_img);
moving_double = im2double(moving_img);

% 4. 修正：多模态配准配置
tform_initial = affine2d(eye(3)); % 初始变换矩阵
[optimizer, metric] = imregconfig('multimodal'); % 关键修改：用'multimodal'模式
% 无需手动设置 metric.Metric！该模式默认就是互信息（MutualInformation）

% 5. 执行刚性配准（平移+旋转+缩放，符合医学图像解剖结构）
registered_ct = imregister(moving_double, ref_double, 'rigid', optimizer, metric);
registered_ct_uint8 = im2uint8(registered_ct); % 转回uint8用于显示

% 6. 像素灰度极大/极小融合
[m, n] = size(ref_img);
fusion_max = zeros(m, n, class(ref_img));
fusion_min = zeros(m, n, class(ref_img));

for i = 1:m
    for j = 1:n
        % 极大法：取两图中灰度绝对值更大的像素
        if abs(ref_double(i,j)) >= abs(registered_ct(i,j))
            fusion_max(i,j) = ref_img(i,j);
        else
            fusion_max(i,j) = registered_ct_uint8(i,j);
        end
        % 极小法：取两图中灰度绝对值更小的像素
        if abs(ref_double(i,j)) >= abs(registered_ct(i,j))
            fusion_min(i,j) = registered_ct_uint8(i,j);
        else
            fusion_min(i,j) = ref_img(i,j);
        end
    end
end

% 7. 显示结果（配准对比+融合结果）
figure('Name','像素灰度极值法融合');
subplot(2,3,1); imshow(ref_img); title('参考MRI');
subplot(2,3,2); imshow(moving_img); title('待配准CT');
subplot(2,3,3); imshow(registered_ct_uint8); title('配准后CT');
subplot(2,3,4); imshow(fusion_max); title('灰度极大法融合');
subplot(2,3,5); imshow(fusion_min); title('灰度极小法融合');