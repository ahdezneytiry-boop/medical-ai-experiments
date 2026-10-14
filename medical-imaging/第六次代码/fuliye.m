%% 傅里叶变换法融合
% 1. 读入图像
ref_img = imread('E:\\ML_Results_Validity\\017.png');   % 参考MRI
moving_img = imread('E:\\ML_Results_Validity\\017 (1).png'); % 待配准CT

% 2. 预处理：转灰度+双精度
ref_img = rgb2gray(ref_img);
moving_img = rgb2gray(moving_img);
ref_double = im2double(ref_img);
moving_double = im2double(moving_img);

% 3. 多模态配准
[optimizer, metric] = imregconfig('multimodal'); % 异模态默认互信息，无额外设置
optimizer.MaximumIterations = 1000; % 唯一兼容参数：频域融合需高精度，提高迭代次数 

% 4. 执行配准+尺寸强制统一
registered_ct = imregister(moving_double, ref_double, 'rigid', optimizer, metric);
registered_ct = imresize(registered_ct, size(ref_double)); % 确保尺寸完全一致
ref_double = imresize(ref_double, size(ref_double)); % 冗余保障

% 5. 傅里叶变换（转至频域）
freq_mri = fft2(ref_double);    % MRI频域系数
freq_ct = fft2(registered_ct);  % 配准后CT频域系数

% 6. 频域加权融合
freq_fusion_eq = 0.5 * freq_mri + 0.5 * freq_ct; % 等权重
freq_fusion_uneq = 0.3 * freq_mri + 0.7 * freq_ct; % 侧重CT高频细节

% 7. 逆变换+去虚部（数值误差处理）
fusion_eq_space = real(ifft2(freq_fusion_eq)); % 取实部，避免虚部干扰显示
fusion_uneq_space = real(ifft2(freq_fusion_uneq));

% 8. 归一化+转uint8（避免图像偏暗/偏亮）
fusion_eq_space = mat2gray(fusion_eq_space); % 归一化到0~1范围
fusion_uneq_space = mat2gray(fusion_uneq_space);
fusion_eq_uint8 = im2uint8(fusion_eq_space);
fusion_uneq_uint8 = im2uint8(fusion_uneq_space);

% 9. 显示结果
figure('Name','傅里叶变换法');
subplot(2,3,1); imshow(ref_img); title('参考MRI');
subplot(2,3,2); imshow(moving_img); title('待配准CT');
subplot(2,3,3); imshow(im2uint8(registered_ct)); title('配准后CT');
subplot(2,3,4); imshow(fusion_eq_uint8); title('频域0.5+0.5融合');
subplot(2,3,5); imshow(fusion_uneq_uint8); title('频域0.3+0.7融合');