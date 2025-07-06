# -*- coding: utf-8 -*-
import pandas as pd
file_path_2 = r'C:\Users\wangy\Desktop\mathc\C\2.xlsx'
file_path_1 = r'C:\Users\wangy\Desktop\mathc\C\1.xlsx'
file_path_3 = r'C:\Users\wangy\Desktop\mathc\C\3.xlsx'

df1 = pd.read_excel(file_path_2, sheet_name='2023年统计的相关数据')
df2 = pd.read_excel(file_path_2, sheet_name='2023年的农作物种植情况')
df3 = pd.read_excel(file_path_1, sheet_name='乡村的现有耕地')
df4 = pd.read_excel(file_path_1, sheet_name='乡村种植的农作物')
df5 = pd.read_excel(file_path_3)

class Crop:
    def __init__(self, crop_type, planting_season, crop_name, planting_land_types,crop_id):
        self.crop_type = crop_type
        self.planting_season = planting_season
        self.crop_name = crop_name
        self.planting_land_types = planting_land_types
        self.crop_id =crop_id
        

crops = []

for i in range(1, 41):  # 从1开始，因为作物编号可能是从1开始的
    # 确保筛选结果是单个值
    crop_name = df4[df4['作物编号'] == i]['作物名称'].iloc[0] if not df4[df4['作物编号'] == i].empty else None
    planting_season = df2[df2['作物编号'] == i]['种植季次'].iloc[0] if not df2[df2['作物编号'] == i].empty else None
    crop_type = df4[df4['作物编号'] == i]['作物类型'].iloc[0] if not df4[df4['作物编号'] == i].empty else None
    planting_land_types = df5[df5['作物编号'] == i]['种植地'].iloc[0] if not df4[df4['作物编号'] == i].empty else None
    crop_id = i

    # 创建Crop对象
    crop = Crop(crop_type, planting_season, crop_name, planting_land_types, crop_id)
    crops.append(crop)

    # 打印信息（如果需要）
    print(
        f"作物名称: {crop_name}, 种植季次: {planting_season}, 作物类型: {crop_type}, 种植耕地: {planting_land_types}, 作物编号: {crop_id}")

    # 现在crops列表包含了所有的Crop对象