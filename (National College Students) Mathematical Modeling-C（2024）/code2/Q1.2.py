import pandas as pd
import numpy as np
import re

def load_data(land_file, crop_land_file, planting_file, crop_file):
    """
    加载所有必要的数据文件。
    """
    try:
        land_data = pd.read_excel(land_file)
        crop_land_data = pd.read_excel(crop_land_file)
        planting_data = pd.read_excel(planting_file)
        crop_data = pd.read_excel(crop_file)
        print("数据加载成功")
        return land_data, crop_land_data, planting_data, crop_data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None

def define_crop_types():
    """
    定义各种作物类型，以便于后续的数据处理和分析。
    """
    return {
        'legume_crops': ['黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆'],
        'grain_crops': ['小麦', '玉米'],
        'vegetable_crops': ['白菜', '生菜', '菠菜', '番茄'],
        'fungi_crops': ['蘑菇', '羊肚菌']
    }

def adjust_parameters_fixed(crop_name, base_cost):
    """
    根据种植成本计算销售单价和亩产量。
    """
    price = 2.5 + 0.0032 * base_cost  # 销售单价公式
    yield_per_mu = 300 + 0.0456 * base_cost  # 亩产量公式
    cost = base_cost  # 种植成本保持不变
    return yield_per_mu, cost, price

def find_best_crop(season_crops, crop_data, plot_area, expected_sales_factor, min_plot_area):
    """
    遍历每个作物，计算收益，并找出最佳作物。
    """
    best_revenue = -np.inf
    best_crop_idx = 0
    for crop_idx in season_crops.index:
        crop_name = season_crops.at[crop_idx, '作物名称']
        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]
        if not crop_data_filtered.empty:
            base_cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']
            yield_per_mu, cost, price = adjust_parameters_fixed(crop_name, base_cost)
            total_production = yield_per_mu * plot_area
            expected_sales = expected_sales_factor * total_production
            surplus_production = max(0, total_production - expected_sales)
            regular_sales = min(total_production, expected_sales)
            revenue = regular_sales * price + surplus_production * (price * 0.5) - cost * plot_area
            if revenue > best_revenue and plot_area >= min_plot_area:
                best_revenue = revenue
                best_crop_idx = crop_data.index[crop_data['作物名称'] == crop_name][0]
    return best_crop_idx, best_revenue

def process_plot_info(land_data, land_types):
    """
    处理地块信息，创建地块名称和面积的DataFrame。
    """
    plot_info = pd.DataFrame()
    for land_type in land_types:
        names = land_data[land_data['地块类型'] == land_type]['地块名称']
        areas = land_data[land_data['地块类型'] == land_type]['地块面积']
        temp_df = pd.DataFrame({'种植地块': names, '地块类型': land_type, '地块面积': areas})
        plot_info = pd.concat([plot_info, temp_df], ignore_index=True)
    return plot_info

def merge_plot_info(planting_data, plot_info):
    """
    将地块信息合并到种植数据中。
    """
    return pd.merge(planting_data, plot_info, on='种植地块', how='left')

def parse_crop_land_data(crop_land_data):
    """
    解析作物适用的地块和季节，返回一个包含所有相关信息的DataFrame。
    """
    data = {
        '作物编号': [],
        '作物名称': [],
        '作物类型': [],
        '地块类型': [],
        '季节': []
    }
    for index, row in crop_land_data.iterrows():
        suitable_lands = str(row['种植耕地']).replace('↵', '').strip()
        if suitable_lands:
            tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)
            for land_type, season in tokens:
                seasons = season.split()
                for season_part in seasons:
                    data['作物编号'].append(row['作物编号'])
                    data['作物名称'].append(row['作物名称'])
                    data['作物类型'].append(row['作物类型'])
                    data['地块类型'].append(land_type)
                    data['季节'].append(season_part)
    return pd.DataFrame(data)

def create_result_table(crop_data_frame):
    """
    根据解析后的数据创建结果表格，并进行排序和去重。
    """
    result_table = crop_data_frame.sort_values(by=['作物编号', '地块类型', '季节'])
    return result_table.drop_duplicates()

def export_result_table(result_table, filename):
    """
    将结果表格导出到Excel文件。
    """
    try:
        result_table.to_excel(filename, index=False)
        print(f"Successfully exported to {filename}")
    except Exception as e:
        print(f"Error exporting data: {e}")

def plan_crops(yearly_plans, years, plot_info, result_table, crop_data, legume_crops, expected_sales_factor, min_plot_area):
    """
    为每个年份规划种植方案，并存储在yearly_plans列表中。
    """
    last_crop_planted = np.empty((len(plot_info), len(years)), dtype=object)
    last_legume_year = np.zeros(len(plot_info))

    for year_idx in years:
        print(f"正在规划 {year_idx} 年的种植方案...")
        year_plan_first_season = np.zeros((len(plot_info), len(crop_data)))
        year_plan_second_season = np.zeros((len(plot_info), len(crop_data)))

        for plot_idx in range(len(plot_info)):
            plot_area = plot_info.at[plot_idx, '地块面积']
            applicable_crops = result_table[result_table['地块类型'] == plot_info.at[plot_idx, '地块类型']]

            # 检查并应用前一年的作物种植情况
            if year_idx > 2024:
                last_crop = last_crop_planted[plot_idx, year_idx - 1]
                applicable_crops = applicable_crops[applicable_crops['作物名称'] != last_crop]

            # 应用豆类作物轮作规则
            if (year_idx - last_legume_year[plot_idx]) >= 3:
                legume_crops_applicable = applicable_crops[applicable_crops['作物名称'].isin(legume_crops)]
                if not legume_crops_applicable.empty:
                    applicable_crops = legume_crops_applicable

            for season in ['第一季', '第二季']:
                season_crops = applicable_crops[applicable_crops['季节'] == season]
                total_planted_area = 0

                if not season_crops.empty:
                    for crop_idx in season_crops.index:
                        crop_name = season_crops.at[crop_idx, '作物名称']
                        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]
                        if not crop_data_filtered.empty:
                            base_cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']
                            yield_per_mu, cost, price = adjust_parameters_fixed(crop_name, base_cost)
                            total_production = yield_per_mu * plot_area
                            expected_sales = expected_sales_factor * total_production
                            surplus_production = max(0, total_production - expected_sales)
                            regular_sales = min(total_production, expected_sales)
                            revenue = regular_sales * price + surplus_production * (price * 0.5) - cost * plot_area

                            if revenue > 0 and total_planted_area < plot_area:
                                planting_area = min(plot_area - total_planted_area, plot_area)
                                if season == '第一季':
                                    year_plan_first_season[plot_idx, crop_idx] = planting_area
                                else:
                                    year_plan_second_season[plot_idx, crop_idx] = planting_area
                                total_planted_area += planting_area

                                # 更新最后种植的作物和豆类作物年份
                                last_crop_planted[plot_idx, year_idx] = crop_name
                                if crop_name in legume_crops:
                                    last_legume_year[plot_idx] = year_idx

        # 将每年的种植方案存储在列表中
        yearly_plans.append((year_plan_first_season, year_plan_second_season))
        print(f"完成 {year_idx} 年的种植方案规划。")


def validate_data_integrity(data):
    """
    验证数据完整性，确保必要的列存在且没有空值。
    """
    required_columns = ['作物名称', '种植地块', '第一季', '第二季']
    for column in required_columns:
        if column not in data.columns:
            print(f"Warning: Missing column {column}")
            return False
        if data[column].isnull().any():
            print(f"Warning: Null values found in column {column}")
            return False
    return True


def log_processing_step(step_description):
    """
    记录处理步骤的日志。
    """
    print(f"Processing step: {step_description}")


def export_plans(yearly_plans, years, plot_info, crop_data):
    """
    将每年的种植方案导出为Excel文件。
    """
    for year_idx, (first_season_plan, second_season_plan) in enumerate(yearly_plans, start=2024):
        log_processing_step(f"Exporting plans for year {year_idx}")

        # 创建 DataFrame
        first_season_df = pd.DataFrame(first_season_plan, columns=crop_data['作物名称'], index=plot_info['种植地块'])
        second_season_df = pd.DataFrame(second_season_plan, columns=crop_data['作物名称'], index=plot_info['种植地块'])

        # 验证数据完整性
        if not validate_data_integrity(first_season_df):
            print(f"Data integrity check failed for first season of year {year_idx}. Skipping export.")
            continue
        if not validate_data_integrity(second_season_df):
            print(f"Data integrity check failed for second season of year {year_idx}. Skipping export.")
            continue

        # 生成文件名
        first_season_filename = f'Optimal_Crop_Plan_Year_{year_idx}_Season_1.xlsx'
        second_season_filename = f'Optimal_Crop_Plan_Year_{year_idx}_Season_2.xlsx'

        # 导出到 Excel
        try:
            first_season_df.to_excel(first_season_filename, index=True)
            second_season_df.to_excel(second_season_filename, index=True)
            print(f"Successfully exported: {first_season_filename}")
            print(f"Successfully exported: {second_season_filename}")
        except Exception as e:
            print(f"Error exporting plans for year {year_idx}: {e}")


def load_data(land_file, crop_land_file, planting_file, crop_file):
    """
    加载所有必要的数据文件。
    """
    try:
        land_data = pd.read_excel(land_file)
        crop_land_data = pd.read_excel(crop_land_file)
        planting_data = pd.read_excel(planting_file)
        crop_data = pd.read_excel(crop_file)
        print("数据加载成功")
        return land_data, crop_land_data, planting_data, crop_data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None


def define_crop_types():
    """
    定义各种作物类型，以便于后续的数据处理和分析。
    """
    return {
        'legume_crops': ['黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆'],
        'grain_crops': ['小麦', '玉米'],
        'vegetable_crops': ['白菜', '生菜', '菠菜', '番茄'],
        'fungi_crops': ['蘑菇', '羊肚菌']
    }


def adjust_parameters_fixed(crop_name, base_cost):
    """
    根据种植成本计算销售单价和亩产量。
    """
    price = 2.5 + 0.0032 * base_cost  # 销售单价公式
    yield_per_mu = 300 + 0.0456 * base_cost  # 亩产量公式
    cost = base_cost  # 种植成本保持不变
    return yield_per_mu, cost, price


def find_best_crop(season_crops, crop_data, plot_area, expected_sales_factor, min_plot_area):
    """
    遍历每个作物，计算收益，并找出最佳作物。
    """
    best_revenue = -np.inf
    best_crop_idx = 0
    for crop_idx in season_crops.index:
        crop_name = season_crops.at[crop_idx, '作物名称']
        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]
        if not crop_data_filtered.empty:
            base_cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']
            yield_per_mu, cost, price = adjust_parameters_fixed(crop_name, base_cost)
            total_production = yield_per_mu * plot_area
            expected_sales = expected_sales_factor * total_production
            surplus_production = max(0, total_production - expected_sales)
            regular_sales = min(total_production, expected_sales)
            revenue = regular_sales * price + surplus_production * (price * 0.5) - cost * plot_area
            if revenue > best_revenue and plot_area >= min_plot_area:
                best_revenue = revenue
                best_crop_idx = crop_data.index[crop_data['作物名称'] == crop_name][0]
    return best_crop_idx, best_revenue


def process_plot_info(land_data, land_types):
    """
    处理地块信息，创建地块名称和面积的DataFrame。
    """
    plot_info = pd.DataFrame()
    for land_type in land_types:
        names = land_data[land_data['地块类型'] == land_type]['地块名称']
        areas = land_data[land_data['地块类型'] == land_type]['地块面积']
        temp_df = pd.DataFrame({'种植地块': names, '地块类型': land_type, '地块面积': areas})
        plot_info = pd.concat([plot_info, temp_df], ignore_index=True)
    return plot_info


def merge_plot_info(planting_data, plot_info):
    """
    将地块信息合并到种植数据中。
    """
    return pd.merge(planting_data, plot_info, on='种植地块', how='left')


def parse_crop_land_data(crop_land_data):
    """
    解析作物适用的地块和季节，返回一个包含所有相关信息的DataFrame。
    """
    data = {
        '作物编号': [],
        '作物名称': [],
        '作物类型': [],
        '地块类型': [],
        '季节': []
    }
    for index, row in crop_land_data.iterrows():
        suitable_lands = str(row['种植耕地']).replace('↵', '').strip()
        if suitable_lands:
            tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)
            for land_type, season in tokens:
                seasons = season.split()
                for season_part in seasons:
                    data['作物编号'].append(row['作物编号'])
                    data['作物名称'].append(row['作物名称'])
                    data['作物类型'].append(row['作物类型'])
                    data['地块类型'].append(land_type)
                    data['季节'].append(season_part)
    return pd.DataFrame(data)


def create_result_table(crop_data_frame):
    """
    根据解析后的数据创建结果表格，并进行排序和去重。
    """
    result_table = crop_data_frame.sort_values(by=['作物编号', '地块类型', '季节'])
    return result_table.drop_duplicates()


def export_result_table(result_table, filename):
    """
    将结果表格导出到Excel文件。
    """
    try:
        result_table.to_excel(filename, index=False)
        print(f"Successfully exported to {filename}")
    except Exception as e:
        print(f"Error exporting data: {e}")


def plan_crops(yearly_plans, years, plot_info, result_table, crop_data, legume_crops, expected_sales_factor,
               min_plot_area):
    """
    为每个年份规划种植方案，并存储在yearly_plans列表中。
    """
    last_crop_planted = np.empty((len(plot_info), len(years)), dtype=object)
    last_legume_year = np.zeros(len(plot_info))

    for year_idx in years:
        print(f"正在规划 {year_idx} 年的种植方案...")
        year_plan_first_season = np.zeros((len(plot_info), len(crop_data)))
        year_plan_second_season = np.zeros((len(plot_info), len(crop_data)))

        for plot_idx in range(len(plot_info)):
            plot_area = plot_info.at[plot_idx, '地块面积']
            applicable_crops = result_table[result_table['地块类型'] == plot_info.at[plot_idx, '地块类型']]

            # 检查并应用前一年的作物种植情况
            if year_idx > 2024:
                last_crop = last_crop_planted[plot_idx, year_idx - 1]
                applicable_crops = applicable_crops[applicable_crops['作物名称'] != last_crop]

            # 应用豆类作物轮作规则
            if (year_idx - last_legume_year[plot_idx]) >= 3:
                legume_crops_applicable = applicable_crops[applicable_crops['作物名称'].isin(legume_crops)]
                

            for season in ['第一季', '第二季']:
                season_crops = applicable_crops[applicable_crops['季节'] == season]
                total_planted_area = 0

                if not season_crops.empty:
                    for crop_idx in season_crops.index:
                        crop_name = season_crops.at[crop_idx, '作物名称']
                        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]
                        if not crop_data_filtered.empty:
                            base_cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']
                            yield_per_mu, cost, price = adjust_parameters_fixed(crop_name, base_cost)
                            total_production = yield_per_mu * plot_area
                            expected_sales = expected_sales_factor * total_production
                            surplus_production = max(0, total_production - expected_sales)
                            regular_sales = min(total_production, expected_sales)
                            revenue = regular_sales * price + surplus_production * (price * 0.5) - cost * plot_area

                            if revenue > 0 and total_planted_area < plot_area:
                                planting_area = min(plot_area - total_planted_area, plot_area)
                                if season == '第一季':
                                    year_plan_first_season[plot_idx, crop_idx] = planting_area
                                else:
                                    year_plan_second_season[plot_idx, crop_idx] = planting_area
                                total_planted_area += planting_area

                                # 更新最后种植的作物和豆类作物年份
                                last_crop_planted[plot_idx, year_idx] = crop_name
                                if crop_name in legume_crops:
                                    last_legume_year[plot_idx] = year_idx

            # 将每年的种植方案存储在列表中
            yearly_plans.append((year_plan_first_season, year_plan_second_season))
            print(f"完成 {year_idx} 年的种植方案规划。")

