import pandas as pd

col_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]

data = pd.read_csv("Real_Datasets/Covertype/covtype.data", names=col_names)

categorical_columns = [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]
for col in categorical_columns:
    data[col] = data[col].astype("object")

data.to_csv("Real_Datasets/Covertype_compressed.csv", index=False)
print(f"预处理完成！Covertype数据集共{len(data)}行，保存路径：Real_Datasets/Covertype_compressed.csv")
