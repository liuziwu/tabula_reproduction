#!/usr/bin/env python
# coding: utf-8

# In[1]:


# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from tabula import Tabula 
import pandas as pd


# In[2]:


data = pd.read_csv("Real_Datasets/Loan_compressed.csv")
# 查看数据基本信息（确认行数、列数、分类列）
print("数据形状：", data.shape)
print("分类列：", data.select_dtypes(include=["object"]).columns.tolist())


# In[3]:


categorical_columns = [
    "Family", "Education", "Securities_Account", 
    "CD_Account", "Online", "CreditCard", "Personal_Loan"
]
model = Tabula(
    llm='distilgpt2', 
    experiment_dir="loan_training", 
    batch_size=32, 
    epochs=100,  # 文档要求Loan数据集训练100个epoch
    categorical_columns=categorical_columns
)


# In[4]:


# Comment this block out to test tabula starting from randomly initialized model.
# Comment this block out when uses tabula_middle_padding
import torch
model.model.load_state_dict(torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False)


# In[5]:


model.fit(data)


# In[6]:


# 保存训练后的模型
torch.save(model.model.state_dict(), "loan_training/model_100epoch.pt")

# 生成合成数据（max_length=100足够覆盖Loan数据集的token长度）
synthetic_data = model.sample(n_samples=5000, max_length=100)
synthetic_data.to_csv("loan_100epoch.csv", index=False)

# 评估F1-score（参考文档：Tabula合成数据F1-score≈0.902）
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df

real_encoded = encode_data(data.copy())
synth_encoded = encode_data(synthetic_data.copy())

X_real, y_real = real_encoded.drop("Personal_Loan", axis=1), real_encoded["Personal_Loan"]
X_synth, y_synth = synth_encoded.drop("Personal_Loan", axis=1), synth_encoded["Personal_Loan"]

# 用合成数据训练，原始数据测试集评估
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_synth, y_synth)
y_pred = rf.predict(X_real.sample(frac=0.2, random_state=42))  # 取1k测试集，匹配文档比例
synth_f1 = f1_score(y_real.sample(frac=0.2, random_state=42), y_pred, average="weighted")
print(f"合成数据F1-score：{synth_f1:.4f}")


# In[ ]:





# In[ ]:




