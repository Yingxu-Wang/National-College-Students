from collections import defaultdict
from datetime import datetime
import pandas as pd

file_path_2 = r'C:\Users\wangy\Desktop\mathc\C\2.xlsx'
file_path_1 = r'C:\Users\wangy\Desktop\mathc\C\1.xlsx'
file_path_3 = r'C:\Users\wangy\Desktop\mathc\C\3.xlsx'

df1 = pd.read_excel(file_path_2, sheet_name='2023年统计的相关数据')
df2 = pd.read_excel(file_path_2, sheet_name='2023年的农作物种植情况')
df3 = pd.read_excel(file_path_1, sheet_name='乡村的现有耕地')
df4 = pd.read_excel(file_path_1, sheet_name='乡村种植的农作物')
df5 = pd.read_excel(file_path_3)

unique_crop_names = df1['作物名称'].unique()

# 这一串代码是我区别售货单价之中的最高的单价和最低的单价的，我们需要吧这两个单价分开来计算
df1[['MinPrice', 'MaxPrice']] = df1['销售单价/(元/斤)'].str.split('-', expand=True).astype(float)

# 下面的代码是我用来确保每一列的数据类型是正确的
df1['亩产量/斤'] = pd.to_numeric(df1['亩产量/斤'], errors='coerce')
df1['种植成本/(元/亩)'] = pd.to_numeric(df1['种植成本/(元/亩)'], errors='coerce')

# 这一串代码是我把同一列之中相同的名字取出来的代码
grouped = df1.groupby(['作物名称'])

# 下面的代码是我用来计算同一组之中每一亩最大的利润和最小的利润的代码
for crop_name, group in grouped:
    group['ProfitRangeMin'] = group['MinPrice'] * group['亩产量/斤'] - group['种植成本/(元/亩)']
    group['ProfitRangeMax'] = group['MaxPrice'] * group['亩产量/斤'] - group['种植成本/(元/亩)']

    print(f"作物名称: {crop_name}")
    print("利润范围: \n")

    # 打印最小利润和最大利润
    for index, row in group.iterrows():
        print(f"{row['ProfitRangeMin']},  {row['ProfitRangeMax']}")
    print("\n")