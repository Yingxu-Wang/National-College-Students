clear;
clc;

% ��ȡExcel�ļ��е�����
data = xlsread('filetwo.xlsx');

% �������ݵ�.mat�ļ��Ա��Ժ�ʹ��
save('filetwo_data', 'data');

% ���ر��������
load('filetwo_data');

% ����������ͳ����
MIN = min(data);  % ����ÿ�е���Сֵ
MAX = max(data);  % ����ÿ�е����ֵ
MEAN = mean(data);  % ����ÿ�еľ�ֵ
MEDIAN = median(data);  % ����ÿ�е���λ��
SKEWNESS = skewness(data);  % ����ÿ�е�ƫ��
KURTOSIS = kurtosis(data);  % ����ÿ�еķ��
STD = std(data);  % ����ÿ�еı�׼��
RESULT = [MIN; MAX; MEAN; MEDIAN; SKEWNESS; KURTOSIS; STD];  % ��ͳ�ƽ�����ϵ�һ��������

% ����Ƥ��ѷ���ϵ��
R = corrcoef(data);

% ����t�ֲ��ĸ����ܶȺ���
x = -4:0.1:4;  % x�᷶Χ
y = tpdf(x, 14);  % ����t�ֲ����ܶ�ֵ�����ɶ�Ϊ14
figure;  % ������ͼ�δ���
plot(x, y, '-');  % ���Ƹ����ܶȺ���
grid on;  % ��ʾ����
hold on;  % ����ͼ���Ա���Ӹ���ͼ��

% ��ʾt�ֲ����ٽ�ֵ
critical_value = tinv(0.975, 14);  % ����95%����ˮƽ���ٽ�ֵ
disp(critical_value);  % ��ʾ�ٽ�ֵ

% ��ͼ�ϻ����ٽ�ֵ��Ӧ�Ĵ�ֱ��
plot([-critical_value, -critical_value], [0, tpdf(-critical_value, 14)], 'r-');
plot([critical_value, critical_value], [0, tpdf(critical_value, 14)], 'r-');

% ����ͼ���Ա��һ������
hold off;
