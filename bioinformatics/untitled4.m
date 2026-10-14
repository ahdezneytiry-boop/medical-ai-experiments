%% ====================== 0. 初始化与文件路径（用户需修改） ======================
clear; clc; close all;
% 输入文件路径（请根据实际文件位置修改）
file_historical = "data/附件2.xlsx";  % 格式：日期×线路×72间隔（含非生产时段0）
file_forecast = "data/附件3.xlsx"; % 格式：日期×线路×预知货量（最后一列为未来1天）

%% ====================== 1. 读取并整理数据（关键步骤） ======================
%% 1.1 读取历史数据（附件2）
% 假设Excel结构：
% - 第1行：日期（如"2023-12-01"）
% - 第2行：线路编码（如"场地3-站点83–0600"）
% - 第3-17行：15天×72个10分钟间隔货量（含非生产时段0）
data_hist = readtable(file_historical, 'ReadRowNames', true);
dates = data_hist.Properties.RowNames; % 日期标签（15行）
line_codes = data_hist.Properties.VariableNames(2:end); % 所有线路编码
num_lines = length(line_codes);
historical_data = cell(1, num_lines); % 存储各线路15×72矩阵

for i = 1:num_lines
    line_code = line_codes{i};
    % 提取该线路15天×72间隔数据（含非生产时段0）
    historical_data{i} = table2array(data_hist(:, line_code)); 
end

%% 1.2 读取预知数据（附件3）
% 假设Excel结构：
% - 前15行为历史预知货量，最后一行为未来1天预知
data_forecast = readtable(file_forecast, 'ReadRowNames', true);
forecast_known = zeros(1, num_lines); % 未来1天预知总货量

for i = 1:num_lines
    line_code = line_codes{i};
    % 最后一行为未来1天预知值（题目中"21点已知未来1天预知货量"）
    forecast_known(i) = data_forecast{end, line_code}; 
end

%% ====================== 2. 定义全局参数（时间间隔映射） ======================
% 生产时段索引（基于题目时间定义）：
% - 每天21:00-次日6:00：54个间隔（索引1-54，对应时间戳21:00, 21:10, ..., 5:50）
% - 每天11:00-14:00：18个间隔（索引55-72，对应时间戳11:00, 11:10, ..., 13:50）
% 非生产时段：6:00-11:00（间隔55-90，超出72范围，故实际非生产时段在72间隔内为0）
% 注意：72个间隔已过滤非生产时段，历史数据中已填充0，无需额外处理

%% ====================== 3. 遍历所有线路进行预测 ======================
results_total = cell(num_lines, 3); % 表1：线路、总货量、时间范围
results_granular = cell(num_lines, 73); % 表2：线路+72个间隔

for line_idx = 1:num_lines
    %% 3.1 数据准备
    line_code = line_codes{line_idx};
    hist_data = historical_data{line_idx}; % 15×72矩阵（已含非生产时段0）
    daily_total = sum(hist_data, 2); % 每日总货量（15×1）
    known_total = forecast_known(line_idx); % 未来1天预知总货量
    
    %% 3.2 ARIMA总货量预测（含预知修正）
    [arima_total, ~] = arima_model(daily_total, known_total);
    
    %% 3.3 LSTM颗粒度预测（10分钟间隔）
    granular_lstm = lstm_model(hist_data);
    
    %% 3.4 结果融合（简单平均，可加权）
    total_pred = (arima_total + sum(granular_lstm)) / 2;
    
    %% 3.5 历史比例分配（核心逻辑，文档1-6段）
    historical_ratio = mean(hist_data, 1); % 72维，含非生产时段0
    historical_ratio = historical_ratio / sum(historical_ratio); % 归一化
    granular_pred = total_pred * historical_ratio'; % 按比例分配
    
    %% 3.6 非负约束（题目要求非生产时段已为0）
    granular_pred(granular_pred < 0) = 0;
    
    %% 3.7 修正总和误差
    if sum(granular_pred) ~= 0
        granular_pred = granular_pred * (total_pred / sum(granular_pred));
    end
    
    %% 3.8 存储结果（按题目时间范围：12月15日14:00-16日14:00）
    % 表1：总货量
    results_total{line_idx, :} = {line_code, total_pred, '2023-12-15 14:00:00至2023-12-16 14:00:00'};
    
    % 表2：72个间隔（生成时间戳，从12月15日14:00开始）
    timestamps = generate_timestamps(); % 自定义函数生成时间戳
    results_granular{line_idx, 1} = line_code;
    results_granular{line_idx, 2:73} = num2cell(granular_pred);
end

%% ====================== 4. 结果输出（生成表1和表2） ======================
% 表1：总货量预测
T1 = cell2table(results_total, 'VariableNames', {'线路编码', '总货量（件）', '时间范围'});
writetable(T1, '结果表1_总货量预测.csv', 'WriteRowNames', false);

% 表2：10分钟颗粒度（含时间戳）
T2 = cell2table(results_granular, 'VariableNames', [{'线路编码'}, generate_timestamps()]);
writetable(T2, '结果表2_10分钟颗粒度.csv', 'WriteRowNames', false);


%% ====================== 5. 子函数定义 ======================
%% 5.1 ARIMA模型（含预知数据修正）
function [total_pred, model] = arima_model(daily_total, known_total)
    % 平稳性检验
    [h, p] = adftest(daily_total);
    d = 0;
    while ~h && d < 2
        daily_total = diff(daily_total);
        [h, p] = adftest(daily_total);
        d = d + 1;
    end
    
    % 自动选择模型
    model = autoarima(daily_total, 'MaxP', 3, 'MaxQ', 3, 'Seasonal', false, 'Display', 'off');
    
    % 预测
    [forecast, ~] = forecast(model, 1);
    if d > 0
        forecast = cumsum([daily_total(end); forecast]);
        total_pred_arima = forecast(2);
    else
        total_pred_arima = forecast;
    end
    
    % 融合预知数据（历史误差率修正，假设历史平均修正系数=1）
    total_pred = known_total; % 实际应计算历史实际/预知比例
end

%% 5.2 LSTM模型（10分钟颗粒度）
function granular = lstm_model(hist_data)
    % 数据整理：15天×72=1080个点
    series = hist_data(:);
    window = 24; % 前24个间隔（4小时）
    X = []; Y = [];
    for i = 1:length(series)-window
        X = cat(3, X, series(i:i+window-1));
        Y = [Y; series(i+window)];
    end
    X = permute(X, [1,3,2]); % LSTM输入格式：[样本, 时间步, 特征]
    
    % 归一化
    [mu, sigma] = deal(mean(series), std(series));
    Xn = (X - mu) / sigma;
    Yn = (Y - mu) / sigma;
    
    % 构建网络
    layers = [
        sequenceInputLayer([window 1])
        lstmLayer(64, 'OutputMode', 'last')
        fullyConnectedLayer(1)
        regressionLayer];
    
    options = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 32, 'Verbose', 0);
    net = trainNetwork(Xn, Yn, layers, options);
    
    % 预测未来72个间隔
    last_window = (series(end-window+1:end) - mu) / sigma;
    granular = zeros(72, 1);
    for i = 1:72
        pred = predict(net, reshape(last_window, 1, window, 1));
        granular(i) = pred * sigma + mu;
        last_window = [last_window(2:end); pred];
    end
end

%% 5.3 生成时间戳（12月15日14:00开始，10分钟间隔）
function timestamps = generate_timestamps()
    start_time = datetime('2023-12-15 14:00:00');
    timestamps = start_time + minutes(0:10:710); % 72个间隔，共12小时
    timestamps = datestr(timestamps, 'yyyy-MM-dd HH:mm:ss');
end