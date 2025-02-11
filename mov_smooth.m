function smoothed_data = mov_smooth(data, min_window, max_window)
    n = length(data);
    smoothed_data = zeros(size(data));
    for i = 1:n
        diff_left = abs(data(i) - data(max(1, i - 1)));
        diff_right = abs(data(min(n, i + 1)) - data(i));
        local_variation = diff_left + diff_right;
        window_size = round(min_window+(max_window - min_window)*(1 - local_variation / max(abs(data(2:end)-data(1:end - 1)))));
        window_size = max(min_window, min(max_window, window_size));
        half_window = floor(window_size / 2);
        start_idx = max(1, i - half_window);
        end_idx = min(n, i + half_window);
        smoothed_data(i)=mean(data(start_idx:end_idx));
    end
end