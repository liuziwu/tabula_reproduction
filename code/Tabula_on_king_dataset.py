#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. 加载King数据集
import pandas as pd
data = pd.read_csv("Real_Datasets/King_compressed.csv")
print("King数据集形状：", data.shape)  # 应与Step 1.2输出一致（如(21613, 21)）

# 2. 定义King数据集分类列（根据Step 1.2列名调整，以下为典型分类列，需按实际列名修正）
categorical_columns = [
    "bedrooms",    # 卧室数量（离散取值：1-10+）
    "bathrooms",   # 浴室数量（离散取值：0.5-8+）
    "floors",      # 楼层数（离散取值：1-3+）
    "waterfront",  # 一线水景（二值：0=无，1=有）
    "view",        # 景观等级（离散取值：0-4）
    "condition",   # 房屋状况（离散取值：1-5）
    "grade",       # 建筑等级（离散取值：1-13）
    "zipcode",     # 邮编（离散取值：多个唯一值）
    "price"        # 房价（标签列，若为分类任务则设为分类列；回归任务可忽略，此处按Tabula分类逻辑设为object）
]

# 3. 强制转换分类列为object类型（确保模型识别）
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype("object")
        print(f"✅ 列 {col} 转换为分类类型")
    else:
        print(f"⚠️  警告：列 {col} 不存在，需核对Step 1.2列名")

# 4. 验证分类列数量
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
print("\\nKing数据集最终分类列数量：", len(cat_cols))  # 参考：9个（上述8个特征+1个标签）
print("分类列列表：", cat_cols)


# In[3]:


# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tabula import Tabula 
import pandas as pd


# In[4]:


data = pd.read_csv("Real_Datasets/King_compressed.csv")
# 查看数据基本信息（确认行数、列数、分类列）
print("数据形状：", data.shape)
print("分类列：", data.select_dtypes(include=["object"]).columns.tolist())


# In[5]:


model = Tabula(
    llm='distilgpt2', 
    experiment_dir="king_training",  # 实验目录（单独命名，避免与Insurance混淆）
    batch_size=16,  # King数据集每行特征较多，减小batch_size防GPU内存不足
    epochs=300,     # 数据量小于Insurance，适当减少epochs（原400→300）
    categorical_columns=categorical_columns  # 传入King数据集分类列
)


# In[6]:


# Comment this block out to test tabula starting from randomly initialized model.
# Comment this block out when uses tabula_middle_padding
import torch
model.model.load_state_dict(torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False)


# In[ ]:


model.fit(data)


# In[6]:


torch.save(model.model.state_dict(), "king_training/model_300epoch.pt")
print("King数据集训练完成，模型保存至：king_training/model_300epoch.pt")



# In[ ]:


# 生成与原数据等量的合成数据（原King数据≈2万行，n_samples设为21613）
synthetic_data = model.sample(
    n_samples=21613, 
    max_length=250  # 与训练时的model_max_length一致，避免编码截断
)

# 保存合成数据（便于后续评估）
synthetic_data.to_csv("king_300epoch.csv", index=False)
print("合成数据生成完成，保存至：king_300epoch.csv")
print("合成数据形状：", synthetic_data.shape)  # 应与原数据一致：(21613, 21)


# In[ ]:




