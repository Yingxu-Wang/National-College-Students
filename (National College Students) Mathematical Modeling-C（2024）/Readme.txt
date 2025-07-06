(National College Students) Mathematical Modeling-C/
├── code/                     # 存放部分代码文件
│   ├── bar_chart_totalMu.py             # 绘制总亩数柱状图
│   ├── bulidstr.py            # 构建字符串相关代码
│   ├── code1.m           # MATLAB代码文件
│   ├── draw_bar_chart.py             # 绘制柱状图
│   ├── profit.py         # 计算利润相关代码
│   └── regression.py             #  回归分析代码 
├── code2/                    #  存放另一部分代码文件
│   ├── GAcode.py      # 遗传算法相关代码
│   ├── Q1.1.py             #  问题1.1的求解代码
│   ├── Q1.2.py             # 问题1.2的求解代码
│   ├── Q2.py             # 问题2的求解代码
│   ├── Q3.py             # 问题3的求解代码
│   └── regression.py           #  回归分析代码
├── Readme


介绍：
本项目针对 2024 年高教社杯全国大学生数学建模竞赛 C 题 "基于农作物种植策略的优化模型分析" 提供了本团队独特的解决方案。
	1. code/bar_chart_totalMu.py
		代码简介
			该脚本用于绘制不同地块类型总亩数的柱状图，直观展示平旱地、梯田、山坡地、水浇地、普通大棚和智慧大棚的面积分布情况。
		参数说明
			--data_file: 可选参数，输入数据文件路径，默认为 './data/land_area.csv'
			--output_file: 可选参数，输出图片保存路径，默认为 './results/land_area_bar_chart.png'
	2. code/bulidstr.py
		代码简介
			提供字符串构建相关功能，主要用于生成模型所需的约束条件字符串、报告文本等。
	3. code/code1.m
		代码简介
			MATLAB 代码文件，主要实现问题一的线性规划模型求解，包含两种销售策略下的优化计算。
	4. code/draw_bar_chart.py
		代码简介
			绘制各类统计柱状图，可用于展示不同作物利润、产量等数据的对比情况。
		参数说明
			--data_type: 数据类型，可选 'profit'（利润）、'yield'（产量）等，默认为 'profit'
			--output_file: 输出图片保存路径，默认为 './results/bar_chart.png'
	5. code/profit.py
		代码简介
			实现利润计算相关功能，支持不同销售策略、价格波动等情况下的利润计算。
	6. code/regression.py
		代码简介
			实现回归分析功能，包括线性回归模型的建立、训练和预测，用于分析亩产量、种植成本和销售单价之间的关系。
	7. code2/GAcode.py
		代码简介
			实现遗传算法相关功能，用于求解农作物种植策略优化问题，包括种群初始化、适应度评估、选择、交叉和变异等操作。
	8. code2/Q1.1.py
		代码简介
			求解问题 1.1，即在假设农作物未来的预期销售量、种植成本、亩产量和销售价格均保持与 2023 年相同，且超过预期销售量的部分滞销并造成浪费情况下的最优农作物种植方案。
	9. code2/Q1.2.py
		代码简介
			求解问题 1.2，即在假设农作物未来的预期销售量、种植成本、亩产量和销售价格均保持与 2023 年相同，且超过预期销售量的部分按 2023 年销售价格的 50% 降价出售情况下的最优农作物种植方案。
		参数说明
			--input_file: 可选参数，输入数据文件路径，默认为 './data/problem1_2_data.csv'
			--output_file: 可选参数，输出结果文件路径，默认为 './results/result1_2.xlsx'
	10. code2/Q2.py
		代码简介
			求解问题 2，即考虑农作物市场存在的不确定性（包括预期销售量的波动、亩产量受气候影响、种植成本增长以及销售价格的变化）情况下的最优农作物种植方案。
		参数说明
			--input_file: 可选参数，输入数据文件路径，默认为 './data/problem2_data.csv'
			--output_file: 可选参数，输出结果文件路径，默认为 './results/result2.xlsx'
	11. code2/Q3.py
		代码简介
			求解问题 3，即在问题 2 的基础上，进一步考虑农作物之间的可替代性和互补性，以及预期销售量、销售价格与种植成本之间的相关性情况下的最优农作物种植策略。
		参数说明
			--input_file: 可选参数，输入数据文件路径，默认为 './data/problem3_data.csv'
			--output_file: 可选参数，输出结果文件路径，默认为 './results/result3.xlsx'
	12. code2/regression.py
		代码简介
			实现回归分析功能，与 code/regression.py 类似，但针对问题 3 的特点进行了优化，主要用于分析农作物之间的相关性和建立多元线性回归模型。


环境配置：
	1. Python 环境
		Python 3.7 及以上版本
		所需 Python 库：
			numpy
			pandas
			matplotlib
			scikit-learn
			xlrd
			openpyxl
	2. MATLAB 环境
			MATLAB R2016b 及以上版本
			优化工具箱 (Optimization Toolbox)


注意事项
	运行代码前请确保已安装所有必要的依赖库
	对于大规模数据或复杂模型，运行时间可能较长，请耐心等待
	部分代码需要根据实际数据格式进行适当调整
	结果文件将保存在 './results/' 目录下，请确保该目录可写
	上述的代码在使用的过程之中需要更改默认的地址。同样的是在处理excel表格的时候需要按照代码之中的参数来进行整理。