import pandas as pd

# 1. 定义Covertype列名（参考UCI文档，共54个特征列+1个目标列）
col_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]

# 2. 读取并设置列名
data = pd.read_csv("Real_Datasets/Covertype/covtype.data", names=col_names)

# 3. 标记分类列（Wilderness_Area_*、Soil_Type_*、Cover_Type）
categorical_columns = [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]
for col in categorical_columns:
    data[col] = data[col].astype("object")

# 4. 保存预处理后的数据（供Notebook调用）
data.to_csv("Real_Datasets/Covertype_compressed.csv", index=False)
print(f"预处理完成！Covertype数据集共{len(data)}行，保存路径：Real_Datasets/Covertype_compressed.csv")
