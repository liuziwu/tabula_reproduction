# %%
# change tabula to tabula_middle_padding to test middle padding method
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tabula_middle_padding import Tabula 
import pandas as pd
import torch

# %%
data = pd.read_csv("/drive1/zhd/Tabula/Real_Datasets/Covertype_compressed.csv")

# %%
#categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(llm='distilgpt2', experiment_dir = "covertype_training", batch_size=32, epochs=30)

# %% [markdown]
# ## In following block, it is important to indicate "conditional_col = data.columns[0]". Otherwise, the generation will use target column as conditional column, which will result in error!

# %%
model.fit(data, conditional_col = data.columns[0])

# %%
import torch
torch.save(model.model.state_dict(), "covertype_training/model_30epoch.pt")

# %%
synthetic_data = model.sample(n_samples=50000, max_length=200)
synthetic_data.to_csv("covertype_30epoch.csv", index=False)

# %%



