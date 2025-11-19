# %%
# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tabula_middle_padding import Tabula 
import pandas as pd
import torch

# %%
data = pd.read_csv("/drive1/zhd/Tabula/Real_Datasets/Adult_compressed.csv")

# %%
#categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(llm='distilgpt2', experiment_dir = "adult_training", batch_size=32, epochs=50)

# %% [markdown]
# ## In following block, it is important to indicate "conditional_col = data.columns[0]". Otherwise, the generation will use target column as conditional column, which will result in error!

# %%
model.fit(data, conditional_col = data.columns[0])

# %%
import torch
torch.save(model.model.state_dict(), "adult_training/model_50epoch.pt")

# %%
synthetic_data = model.sample(n_samples=6000)
synthetic_data.to_csv("adult_50epoch.csv", index=False)

# %%



