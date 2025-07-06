# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
file_path_2 = r'C:\Users\wangy\Desktop\mathc\C\2.xlsx'
file_path_1 = r'C:\Users\wangy\Desktop\mathc\C\1.xlsx'
file_path_3 = r'C:\Users\wangy\Desktop\mathc\C\3.xlsx'

df1 = pd.read_excel(file_path_2, sheet_name='2023年统计的相关数据')
df2 = pd.read_excel(file_path_2, sheet_name='2023年的农作物种植情况')
df3 = pd.read_excel(file_path_1, sheet_name='乡村的现有耕地')
df4 = pd.read_excel(file_path_1, sheet_name='乡村种植的农作物')
df5 = pd.read_excel(file_path_3)
# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 确保列名正确，这里假设列名已经是'地块类型'和'地块面积/亩'
# 如果列名不同，请替换为实际的列名
grouped = df3.groupby('地块类型')['地块面积/亩'].sum().reset_index()

# 绘制条形图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.bar(grouped['地块类型'], grouped['地块面积/亩'], color='skyblue')
plt.xlabel('地块类型', fontsize=14)
plt.ylabel('地块面积/亩', fontsize=14)
plt.title('地块类型与地块面积总和', fontsize=16)
plt.xticks(rotation=45, ha="right")  # 旋转x轴标签，以便更好地显示
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()