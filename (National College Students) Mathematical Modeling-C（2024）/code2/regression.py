import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path_2 = r'C:\Users\wangy\Desktop\mathc\C\2.xlsx'
df1 = pd.read_excel(file_path_2, sheet_name='2023年统计的相关数据')

# 处理销售单价列，计算范围的平均值
split_data = df1['销售单价/(元/斤)'].str.split('-', expand=True)
df1['销售单价/(元/斤)'] = (split_data[0].astype(float) + split_data[1].astype(float)) / 2

# 确保种植成本是数值类型
df1['种植成本/(元/亩)'] = pd.to_numeric(df1['种植成本/(元/亩)'], errors='coerce')

# 准备数据
X = df1[['种植成本/(元/亩)']]  # 特征
y = df1['销售单价/(元/斤)']  # 目标变量

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个 imputer 对象，使用均值填充缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 创建一个线性回归模型
model = LinearRegression()

# 创建一个管道，将 imputer 和模型串联起来
pipeline = make_pipeline(imputer, model)

# 训练模型
pipeline.fit(X_train, y_train)

# 预测测试集
y_pred = pipeline.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 绘制回归线和数据点
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')

# 显示图形
plt.xlabel('种植成本/(元/亩)')
plt.ylabel('销售单价/(元/斤)')
plt.title('回归分析：种植成本 vs 销售单价')
plt.legend()
plt.show()