import pandas as pd
import numpy as np
import pygad

# 读取 Excel 文件
land = pd.read_excel('C:\\Users\\wangy\\Desktop\\数学建模C题\\pretreatment\\landinfo.xlsx')
crop = pd.read_excel('C:\\Users\\wangy\\Desktop\\数学建模C题\\pretreatment\\cropinfo.xlsx')
sale = pd.read_excel('C:\\Users\\wangy\\Desktop\\数学建模C题\\pretreatment\\saleinfo.xlsx')

# 确保 crop DataFrame 中的 '总销售量' 列是数值类型
crop['总销售量'] = pd.to_numeric(crop['总销售量'], errors='coerce')

# 确保 sale DataFrame 中的相关列是数值类型
sale['亩产量/斤'] = pd.to_numeric(sale['亩产量/斤'], errors='coerce')
sale['种植成本/(元/亩)'] = pd.to_numeric(sale['种植成本/(元/亩)'], errors='coerce')

def calculate_revenue(solution, land, crop, sale):
    total_cost, total_price = 0, 0
    for season in range(7):
        yields = np.zeros(len(crop))
        for idx, (l, c) in enumerate(zip(range(len(land)), solution[season * len(land): (season + 1) * len(land)])):
            crop_data = sale[(sale['作物编号'] == c + 1) & (sale['地块类型'] == land.iloc[l, 1] + 1)]
            if not crop_data.empty:
                yield_per_mu, cost_per_mu = crop_data['亩产量/斤'].values[0], crop_data['种植成本/(元/亩)'].values[0]
                yields[c] = land.iloc[l, 1] * yield_per_mu
                total_cost += land.iloc[l, 1] * cost_per_mu

        for crop_idx in range(len(crop)):
            crop_data = sale[sale['作物编号'] == crop_idx + 1]
            if not crop_data.empty:
                price = crop_data['种植成本/(元/亩)'].values[0]
                sales_volume = crop.iloc[crop_idx, 2]
                total_price += (yields[crop_idx] if pd.notnull(yields[crop_idx]) and yields[crop_idx] <= sales_volume else sales_volume) * price

    return total_price - total_cost

def check_constraints(solution, land, crop, sale):
    violations = np.zeros(3)
    for season in range(7):
        for idx, (l, c) in enumerate(zip(range(len(land)), solution[season * len(land): (season + 1) * len(land)])):
            planting_field = crop.loc[c - 1, '种植耕地']
            land_type = land.iloc[l, 1]
            if planting_field == 1 and not (1 <= land_type <= 3):
                violations[2] += 1
            elif planting_field == 2 and land_type != 4:
                violations[2] += 1
            elif planting_field == 3 and not (4 <= land_type <= 6 or land_type == 9):
                violations[2] += 1
            elif planting_field == 4 and land_type != 7:
                violations[2] += 1
            elif planting_field == 5 and land_type != 8:
                violations[2] += 1
    return violations

# 定义遗传算法参数
num_variables = 7 * len(land)
ga_params = {
    'num_generations': 1000,
    'num_parents_mating': 5,
    'fitness_func': lambda ga_instance, solution, solution_idx: -calculate_revenue(solution, land, crop, sale) - np.sum(np.maximum(0, check_constraints(solution, land, crop, sale))),
    'sol_per_pop': 10,
    'num_genes': num_variables,
    'init_range_low': 1,
    'init_range_high': len(crop) + 1,
    'mutation_percent_genes': 10,
    'gene_type': int,
    'parent_selection_type': "sss",
    'crossover_type': "single_point",
    'mutation_type': "random",
    'on_generation': lambda x: print(f"Generation {x.generations_completed}: Best Fitness {x.best_solution()[1]}")
}

ga_instance = pygad.GA(**ga_params)
ga_instance.run()

# 获取并打印结果
best_solution, best_fitness, _ = ga_instance.best_solution()
print(f"最优解: {best_solution}")
print(f"最优目标函数值: {best_fitness}")