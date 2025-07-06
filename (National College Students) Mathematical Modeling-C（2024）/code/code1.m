clear;
clc;

% 读取Excel文件中的数据
data = xlsread('filetwo.xlsx');

% 保存数据到.mat文件以便以后使用
save('filetwo_data', 'data');

% 加载保存的数据
load('filetwo_data');

% 计算描述性统计量
MIN = min(data);  % 计算每列的最小值
MAX = max(data);  % 计算每列的最大值
MEAN = mean(data);  % 计算每列的均值
MEDIAN = median(data);  % 计算每列的中位数
SKEWNESS = skewness(data);  % 计算每列的偏度
KURTOSIS = kurtosis(data);  % 计算每列的峰度
STD = std(data);  % 计算每列的标准差
RESULT = [MIN; MAX; MEAN; MEDIAN; SKEWNESS; KURTOSIS; STD];  % 将统计结果整合到一个矩阵中

% 计算皮尔逊相关系数
R = corrcoef(data);

% 绘制t分布的概率密度函数
x = -4:0.1:4;  % x轴范围
y = tpdf(x, 14);  % 计算t分布的密度值，自由度为14
figure;  % 创建新图形窗口
plot(x, y, '-');  % 绘制概率密度函数
grid on;  % 显示网格
hold on;  % 保持图像，以便添加更多图层

% 显示t分布的临界值
critical_value = tinv(0.975, 14);  % 计算95%置信水平的临界值
disp(critical_value);  % 显示临界值

% 在图上绘制临界值对应的垂直线
plot([-critical_value, -critical_value], [0, tpdf(-critical_value, 14)], 'r-');
plot([critical_value, critical_value], [0, tpdf(critical_value, 14)], 'r-');

% 保持图像，以便进一步分析
hold off;
