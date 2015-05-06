function x_filt = myfilter(x,window_size)
clc
if mod(window_size,2) == 0
    window_size = window_size + 1;
end
x_filt = zeros(size(x));
len_filt = length(x_filt);
half_window = floor(window_size/2);
B = zeros(1,window_size);
alpha = 1/half_window^2;
for i = 1:half_window+1
    B(i) = alpha*(i-1);
end
for i = window_size:-1:window_size-half_window
    B(i) = -alpha*(i-window_size);
end

for i = 1:half_window
    red_window_size = half_window + i;
    B_red = zeros(1,red_window_size);
    K = i;
    W = red_window_size;
    alpha = 2/(-2*K^2 + W^2 + 4*K - 2*W -1);
    for j = 1:i
        B_red(j) = alpha*j + alpha*(W-2*K);
    end
    for j = i+1:red_window_size
        B_red(j) = -alpha*(j) + alpha*(W);
    end
    x_filt(i) =  B_red*x(1:red_window_size);
    B_red = B_red(end:-1:1);
    x_filt(end-i+1) =  B_red*x(end-red_window_size+1:end);
end

for i = half_window + 1 :len_filt-half_window
    x_filt(i) = B*x(i-half_window:i+half_window);
end