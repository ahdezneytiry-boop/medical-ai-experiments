%% 像素加权法融合
% 1. 读入图像
ref_img = imread('E:/ML_Results_Validity/017.png');   % 参考MRI
moving_img = imread('E:/ML_Results_Validity/017 (1).png'); % 待配准CT

% 2. 预处理：转灰度+双精度
ref_img = rgb2gray(ref_img);
moving_img = rgb2gray(moving_img);
ref_double = im2double(ref_img);
moving_double = im2double(moving_img);

% 3. 修正：多模态配准配置
tform_initial = affine2d(eye(3));
[optimizer, metric] = imregconfig('multimodal'); % 关键修改：异模态用'multimodal'
optimizer.MaximumIterations = 500; % 优化迭代次数

% 4. 执行配准
registered_ct = imregister(moving_double, ref_double, 'rigid', optimizer, metric);
registered_ct_uint8 = im2uint8(registered_ct);

% 5. 确保配准后尺寸与参考图一致
if size(registered_ct_uint8) ~= size(ref_img)
    registered_ct_uint8 = imresize(registered_ct_uint8, size(ref_img));
    registered_ct = im2double(registered_ct_uint8); % 同步更新双精度版本用于融合
end

% 6. 加权融合
fusion_eq = 0.5 * ref_double + 0.5 * registered_ct; % 等权重
fusion_uneq = 0.3 * ref_double + 0.7 * registered_ct; % 侧重CT细节
fusion_eq_uint8 = im2uint8(fusion_eq);
fusion_uneq_uint8 = im2uint8(fusion_uneq);

% 7. 显示结果
figure('Name','像素加权法融合');
subplot(2,3,1); imshow(ref_img); title('参考MRI');
subplot(2,3,2); imshow(moving_img); title('待配准CT');
subplot(2,3,3); imshow(registered_ct_uint8); title('配准后CT');
subplot(2,3,4); imshow(fusion_eq_uint8); title('0.5+0.5等权重融合');
subplot(2,3,5); imshow(fusion_uneq_uint8); title('0.3+0.7侧重CT融合');