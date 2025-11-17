#!/usr/bin/env python
# coding: utf-8

# In[1]:


# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from tabula import Tabula 
import pandas as pd


# In[2]:


data = pd.read_csv("Real_Datasets/Insurance_compressed.csv")


# In[3]:


categorical_columns = ["sex", "children", "sm", "region"]
model = Tabula(llm='distilgpt2', experiment_dir = "insurance_training", batch_size=32, epochs=400, categorical_columns = categorical_columns)


# In[4]:


# Comment this block out to test tabula starting from randomly initialized model.
# Comment this block out when uses tabula_middle_padding
import torch
model.model.load_state_dict(torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False)


# In[5]:


model.fit(data)


# In[8]:


import torch
torch.save(model.model.state_dict(), "insurance_training/model_400epoch.pt")


# In[9]:


synthetic_data = model.sample(n_samples=1338)
synthetic_data.to_csv("insurance_400epoch.csv", index=False)


# In[ ]:




