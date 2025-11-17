#!/usr/bin/env python
# coding: utf-8

# In[1]:


# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tabula import Tabula 
import pandas as pd


# In[2]:


data = pd.read_csv("Real_Datasets/Adult_compressed.csv")


# In[3]:


# 1. 指定Adult数据集的分类列（共9个，参考文档和数据查看结果）
categorical_columns = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country", "income"
]

# 2. 初始化Tabula模型（关键调整：epochs=50，因Adult是大数据集，参考文档LLM-based训练设置）
# 文档说明：LLM-based模型在大数据集上训练50 epochs，小数据集（如Insurance）用400 epochs
model = Tabula(
    llm='distilgpt2', 
    experiment_dir="adult_training",  # 实验目录单独命名，避免与Insurance冲突
    batch_size=32,  # 保持原批次大小，若GPU内存不足可改为16
    epochs=50,  # 核心调整：Adult为大数据集，epochs设为50
    categorical_columns=categorical_columns
)


# In[4]:


# Comment this block out to test tabula starting from randomly initialized model.
# Comment this block out when uses tabula_middle_padding
import torch
model.model.load_state_dict(torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False)


# In[6]:


model.fit(data)


# In[7]:


# 保存训练后的模型（目录与实验目录一致）
import torch
torch.save(model.model.state_dict(), "adult_training/model_50epoch.pt")

# 生成合成数据（n_samples建议与原始数据行数接近，max_length设为150）
synthetic_data = model.sample(n_samples=45000, max_length=150)  # 45000≈预处理后的数据量

# 保存合成数据
synthetic_data.to_csv("adult_50epoch.csv", index=False)
print("合成数据生成完成！保存路径：adult_50epoch.csv")


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. 处理评估数据（对分类列编码）
def encode_data(df):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df

# 2. 编码原始数据和合成数据
real_data_encoded = encode_data(data.copy())
synth_data_encoded = encode_data(synthetic_data.copy())

# 3. 分割特征和目标（目标列为income）
X_real, y_real = real_data_encoded.drop("income", axis=1), real_data_encoded["income"]
X_synth, y_synth = synth_data_encoded.drop("income", axis=1), synth_data_encoded["income"]

# 4. 分割训练集和测试集（用原始数据的测试集评估）
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42
)

# 5. 训练随机森林模型（参考文档评估方法）
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 用原始训练集训练，测试集评估
rf.fit(X_real_train, y_real_train)
y_pred_real = rf.predict(X_real_test)
real_f1 = f1_score(y_real_test, y_pred_real, average="weighted")

# 用合成数据训练，原始测试集评估（核心评估合成数据效用）
rf.fit(X_synth, y_synth)
y_pred_synth = rf.predict(X_real_test)
synth_f1 = f1_score(y_real_test, y_pred_synth, average="weighted")

# 输出评估结果
print(f"原始数据F1-score：{real_f1:.4f}（参考文档：0.723）")
print(f"合成数据F1-score：{synth_f1:.4f}（参考文档：0.740）")
print("复现成功：若合成数据F1-score≥0.73，即接近文档结果")


# In[ ]:




