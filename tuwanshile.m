%00000000000000000000000000000000000000
%             输入参数
%00000000000000000000000000000000000000
v = 5;            % 单位mm/s，电机运行速度
t = 80;           % 单位ms，光谱仪积分时间
x_distance = 30;  % 单位mm，电机x轴总位移距离
points_per_column = x_distance * 1000 / (v * t);  % 每一列点的个数
%11111111111111111111111111111111111111
%            数据预处理
%11111111111111111111111111111111111111
%% -----------加载数据-----------
load('Av1500_5_80_gt.mat');
x = bochang;
load('Av1500_5_80.mat');

y1 = 1:size(data, 2);
z1 = data;
colIndices = 1:100:size(data, 2);
numCols = length(colIndices);
y = zeros(size(data, 1), numCols);
z = cell(1, numCols);
denoised_z = cell(1, numCols);
%% -------自适应平滑预处理--------
min_window = 3;
max_window = 9;
for j = 1:size(z1, 2)
    z1(:, j)=mov_smooth(z1(:, j), min_window, max_window);
end
%% -------小波去噪-----------
for j = 1:size(z1, 2)
    [C, L] = wavedec(z1(:, j), 13, "sym4");
    cA_13 = C(1 : L(1));
    startIndex = L(1)+1;
    cD = cell(13, 1);
    for level = 2:1:14
        endPoint = startIndex+(L(level)-1);
        cD{15 - level, 1}=C(startIndex : endPoint);
        startIndex = endPoint + 1;
    end
    miu = 6;
    delta = 0.1;
    for level = 1:1:13
        sigma = 2.5;
        lambda_j = (sigma * sqrt(2 * log(length(z1(:, j)))))/log(level + 1);
        for index = 1:1:length(cD{level, 1})
            cD{level, 1}(index)=WAVET(cD{level, 1}(index), miu, delta, lambda_j);
        end
    end

    new_C = [];
    new_C = cat(1, new_C, cA_13);
    for level = 13:-1:1
        wj = cD{level, 1};
        new_C = cat(1, new_C, wj);
    end
    denoised_z{j}=waverec(new_C, L, "sym4");% 小波重构
end
%% -------二次小波去噪-----------
for j = 1:size(z1, 2)
    [C, L] = wavedec(denoised_z{j}, 13, "sym4");
    cA_13 = C(1 : L(1));
    startIndex = L(1)+1;
    cD = cell(13, 1);
    for level = 2:1:14
        endPoint = startIndex+(L(level)-1);
        cD{15 - level, 1}=C(startIndex : endPoint);
        startIndex = endPoint + 1;
    end
    miu = 6;
    delta = 0.1;
    for level = 1:1:13
        sigma = 2.5;
        lambda_j = (sigma * sqrt(2 * log(length(denoised_z{j}))))/log(level + 1);
        for index = 1:1:length(cD{level, 1})
            cD{level, 1}(index)=WAVET(cD{level, 1}(index), miu, delta, lambda_j);
        end
    end
    new_C = [];
    new_C = cat(1, new_C, cA_13);
    for level = 13:-1:1
        wj = cD{level, 1};
        new_C = cat(1, new_C, wj);
    end
    denoised_z{j}=waverec(new_C, L, "sym4");% 第二次小波重构
end
%% ---------低通滤波-----------
fc = 0.1; % 截止频率
fs = 1; % 采样频率
[b, a] = butter(4, fc / (fs / 2));
for j = 1:size(z1, 2)
    denoised_z{j}=filter(b, a, denoised_z{j});
end

%% -----------提取抽样数据-----------
for i = 1:numCols
    colIndex = colIndices(i);
    y(:, i) = z1(:, colIndex);
    z{i} = z1(:, colIndex);
end
%% -------抽样绘制原始数据--------
new_y = 1:numCols;
figure;
for i = 1:numCols
    plot3(x, new_y(i) * ones(size(x)), z{i});
    hold on;
end
hold off;
xlabel('X_波长');
ylabel('Y_列数');
zlabel('Z_值');
title('原始抽样图');

%% -------抽样绘制去噪后数据--------
figure;
for i = 1:numCols
    plot3(x, new_y(i) * ones(size(x)), denoised_z{colIndices(i)});
    hold on;
end
hold off;
xlabel('X_波长');
ylabel('Y_列数');
zlabel('Z_值');
title('去噪后抽样图');
%22222222222222222222222222222222222222
%%               形成图像
%22222222222222222222222222222222222222
%% -----------定量分析-------------
% 将数据传输到 GPU
x_gpu = gpuArray(x);
denoised_z_matrix = cell2mat(denoised_z);% 将 denoised_z 中的数据存储为矩阵或向量
z1_gpu = gpuArray(denoised_z_matrix);

% 计算每列点的数量
points_per_column = x_distance * 1000 / (v * t);  % 每一列点的个数

% 元素及其对应的波长范围
elements = {'Fe', 'Co', 'Mn', 'Cr', 'Mg'};
wavelengths = [238.58, 252.09, 279.97, 288.57, 309.63];
cr_wavelengths = [288.57, 285.56]; % 铬的两个特征波长
tolerance = 0.01; % 波长范围的容忍度
element_contents = zeros(length(elements), 1);% 存储元素含量


% 颜色映射表，根据元素选择不同的颜色梯度
color_mapping = containers.Map({ 'Fe', 'Co', 'Mn', 'Cr', 'Mg' },...
                           { [255/255, 220/255, 220/255; 110/255, 0/255, 0/255],... % Red
                             [211/255, 246/255, 204/255; 015/255, 063/255, 018/255],... % Green
                             [255/255, 255/255, 224/255; 255/255, 140/255, 0/255],... % Yellow
                             [220/255, 220/255, 220/255; 50/255, 50/255, 50/255],... % Grey
                             [222/255, 184/255, 135/255; 139/255, 69/255, 19/255] }); % Brown


for element_index = 1:length(elements)
    target_wavelengths = wavelengths(element_index);
    if strcmp(elements{element_index}, 'Cr')
        target_wavelengths = cr_wavelengths; % 对于 Cr 使用两个特征波长
    end
    peak_values = gpuArray.zeros(1, size(denoised_z_matrix, 2));
    for target_wavelength = target_wavelengths
        lower_bound = target_wavelength - tolerance;
        upper_bound = target_wavelength + tolerance;
        target_indices = find(x_gpu >= lower_bound & x_gpu <= upper_bound);
        for col = 1:size(denoised_z_matrix, 2)
            peak_value = max(z1_gpu(target_indices, col));
            peak_values(col) = peak_values(col) + peak_value; % 累加峰值
        end
    end
    % 计算平均峰值作为元素含量的估计
    element_contents(element_index) = mean(gather(peak_values)); 
    
    % 绘制图像
    figure;
    hold on;
    all_plot_x = [];
    all_plot_y = [];
    for y1_index = 1:points_per_column:size(y1, 2)
        col_index = (y1_index - 1) / points_per_column + 1;
        row_index = 1;
        actual_indices = y1_index:y1_index + points_per_column - 1;
        actual_indices = actual_indices(actual_indices <= length(y1));
        % 将 x 和 y 数据存入 all_plot_x 和 all_plot_y
        plot_x = repmat(col_index, [1, length(actual_indices)]);
        plot_y = row_index:row_index + length(actual_indices) - 1;
        all_plot_x = [all_plot_x, gather(plot_x)];  % 将数据收集到一个数组中
        all_plot_y = [all_plot_y, gather(plot_y)];  % 将数据收集到一个数组中
    end
    % 选择颜色梯度
    color_gradient = color_mapping(elements{element_index});
    % 定义颜色渐变的数量
    num_colors = 10;
    % 使用线性插值生成颜色渐变
    colors = interp1([1, 2], color_gradient, linspace(1, 2, num_colors), 'linear', 'extrap');
    % 选择 colormap 并获取 RGB 值
    cmap = colors;  % 使用自定义的颜色渐变作为 colormap
    max_peak = max(peak_values);
    normalized_peaks = peak_values / max_peak;% 归一化峰值
    normalized_peaks = max(normalized_peaks, 1e-5);
    rgb_indices = round(normalized_peaks * (num_colors - 1)) + 1;  % 归一化值映射到 [1, num_colors]
    rgb_indices = min(max(rgb_indices, 1), num_colors);  % 确保索引范围在 [1, num_colors]
    rgb_values = cmap(rgb_indices, :);  % 获取对应的 RGB 颜色
    % 处理峰值小于一定值的情况
    peak_mask = peak_values < max_peak/20;
    rgb_values(peak_mask, :) = repmat([173/255 216/255 230/255], [sum(peak_mask), 1]);    % 遍历所有要绘制的点
    for i = 1:length(all_plot_x)
        col_index = all_plot_x(i);
        row_index = all_plot_y(i);
        % 根据列索引和每列点数计算对应的峰值索引
        peak_index = (col_index - 1) * points_per_column + row_index;
        % 获取对应的 RGB 颜色
        color = rgb_values(peak_index, :);
        % 绘制方块并填充颜色
        rectangle('Position', [col_index - 0.5, row_index - 0.5, 1, 1], 'FaceColor', color, 'EdgeColor', 'none');
    end
    hold off;
    % 设置图形属性
    axis equal
    xlim([0, size(denoised_z_matrix, 2) / points_per_column]);
    ylim([0, points_per_column]);
    xlabel('横向');
    ylabel('纵向');
    title([elements{element_index} ' 的相对含量特征图']);
    % 绘制 RGB 标尺
    caxis([0, 1]);
    colorbar;
    colormap(cmap);
end


% 对元素含量进行排序
[sorted_contents, sorted_indices] = sort(element_contents, 'descend');
% 显示排序结果
disp('元素及其相对含量（从高到低）:');
for i = 1:length(elements)
    fprintf('%s\n', elements{sorted_indices(i)});
end