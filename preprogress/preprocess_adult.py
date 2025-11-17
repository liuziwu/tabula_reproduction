import pandas as pd
import numpy as np

# 1. 定义Adult数据集列名（参考UCI说明文件）
adult_columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# 2. 读取训练集和测试集（原始数据用逗号分隔，无表头）
train_data = pd.read_csv(
    "Real_Datasets/Adult/adult.data",
    names=adult_columns,
    sep=", ",  # 注意原始数据是“逗号+空格”分隔
    engine="python",
    na_values="?"  # 原始数据中缺失值用“?”表示
)
test_data = pd.read_csv(
    "Real_Datasets/Adult/adult.test",
    names=adult_columns,
    sep=", ",
    engine="python",
    na_values="?",
    skiprows=1  # 测试集第一行是无关说明，跳过
)

# 3. 合并训练集和测试集（Tabula默认用全量数据训练，后续可自行分割）
adult_data = pd.concat([train_data, test_data], ignore_index=True)

# 4. 处理缺失值（简单删除，也可按文档方法填充，不影响复现核心流程）
adult_data = adult_data.dropna()

# 5. 简化目标列（income）：将“>50K”“>50K.”统一为“>50K”，“<=50K”“<=50K.”统一为“<=50K”
adult_data["income"] = adult_data["income"].str.replace(".", "", regex=False)

# 6. 保存为CSV文件（供Notebook调用）
adult_data.to_csv("Real_Datasets/Adult_compressed.csv", index=False)
print(f"预处理完成！Adult数据集共{len(adult_data)}行，保存路径：Real_Datasets/Adult_compressed.csv")
