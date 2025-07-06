from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
file_path_2 = r'C:\Users\wangy\Desktop\mathc\C\2.xlsx'
file_path_1 = r'C:\Users\wangy\Desktop\mathc\C\1.xlsx'
file_path_3 = r'C:\Users\wangy\Desktop\mathc\C\3.xlsx'

df1 = pd.read_excel(file_path_2, sheet_name='2023年统计的相关数据')
# ... 其他DataFrame的读取可以保持不变 ...

# 分离销售单价
df1[['MinPrice', 'MaxPrice']] = df1['销售单价/(元/斤)'].str.split('-', expand=True).astype(float)

# 转换数据类型
df1['亩产量/斤'] = pd.to_numeric(df1['亩产量/斤'], errors='coerce')
df1['种植成本/(元/亩)'] = pd.to_numeric(df1['种植成本/(元/亩)'], errors='coerce')

# 计算利润
grouped = df1.groupby(['作物名称'])

# 初始化总的最大和最小利润
total_max_profits = []
total_min_profits = []

# 遍历每种作物并计算利润
total_profits = defaultdict(lambda: [0, 0])  # 默认为0

# 遍历每种作物并计算利润
for crop_name, group in grouped:
        if not group.empty:  # 确保组不为空
                group['ProfitRangeMin'] = group['MinPrice'] * group['亩产量/斤'] - group['种植成本/(元/亩)']
                group['ProfitRangeMax'] = group['MaxPrice'] * group['亩产量/斤'] - group['种植成本/(元/亩)']

                total_max_profit = group['ProfitRangeMax'].sum()
                total_min_profit = group['ProfitRangeMin'].sum()

                total_profits[crop_name] = [total_min_profit, total_max_profit]

        # 准备绘图数据
crops = list(total_profits.keys())
total_max_profits = [total_profits[crop][1] for crop in crops]
total_min_profits = [total_profits[crop][0] for crop in crops]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False

# 绘制条形图
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(crops))

plt.bar(index, total_max_profits, bar_width, label='最大利润', alpha=0.7, color='b')
plt.bar(index + bar_width, total_min_profits, bar_width, label='最小利润', alpha=0.7, color='r')

plt.xlabel('cropname')
plt.ylabel('profit')
plt.title('max and min profit/pre MU')
plt.xticks(index + bar_width / 2, crops, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()