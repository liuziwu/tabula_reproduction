import os
import pandas as pd
import torch
from tabula import Tabula 
from tabula.tabula_start import RandomStart  # 保留采样修复所需导入

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# 步骤1：加载压缩数据集（恢复原读取逻辑，不强制匹配完整列名）
data = pd.read_csv("Real_Datasets/Covertype_compressed.csv")
# 直接用压缩文件自带的列名（避免列名不匹配导致的问题）
all_columns = data.columns.tolist()
print(f"压缩数据集列名：{all_columns}")
print(f"压缩数据集原始行数：{len(data)}")  # 验证是否恢复之前的样本量


# 步骤2：定义分类列（仅包含数据中实际存在的列）
# 先筛选出数据中真实存在的分类列（避免列名不存在导致的转换错误）
categorical_columns = [
    "Wilderness_Area_0", "Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3"
] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]
# 过滤掉数据中不存在的列（关键：避免后续转换报错）
existing_cat_cols = [col for col in categorical_columns if col in data.columns]
print(f"实际存在的分类列：{existing_cat_cols}，共{len(existing_cat_cols)}个")


# 步骤3：温和处理分类列（避免强制转换导致数据丢失）
for col in existing_cat_cols:
    # 仅在列是数值型时转换为object（避免非数值列转换出错）
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].astype("object", errors="ignore")  # 允许转换失败，保留原数据
print(f"处理后数据集有效行数：{len(data)}")  # 确认是否有大量行被过滤


# 步骤4：初始化模型（保持原参数）
model = Tabula(
    llm='distilgpt2', 
    experiment_dir="covertype_training", 
    batch_size=32, 
    epochs=30, 
    categorical_columns=existing_cat_cols  # 用实际存在的分类列
)


# 步骤5：加载预训练模型（保留宽松加载）
model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"),
    strict=False  # 避免结构差异报错
)


# 步骤6：训练模型
model.fit(data)


# 步骤7：保存训练后的模型（路径与实验目录一致）
torch.save(model.model.state_dict(), "covertype_training/model_30epoch.pt")


# 步骤8：生成合成数据（保留采样修复）
start_generator = RandomStart(
    all_columns=all_columns,  # 用压缩文件实际列名
    tokenizer=model.tokenizer
)
synthetic_data = model.sample(
    n_samples=50000, 
    max_length=200,
    start_col_generator=start_generator
)
synthetic_data.to_csv("covertype_30epoch.csv", index=False)

print("代码运行完成，合成数据已保存")