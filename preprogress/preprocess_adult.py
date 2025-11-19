import pandas as pd
import numpy as np

adult_columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

train_data = pd.read_csv(
    "Real_Datasets/Adult/adult.data",
    names=adult_columns,
    sep=", ",  
    engine="python",
    na_values="?" 

test_data = pd.read_csv(
    "Real_Datasets/Adult/adult.test",
    names=adult_columns,
    sep=", ",
    engine="python",
    na_values="?",
    skiprows=1  
)

adult_data = pd.concat([train_data, test_data], ignore_index=True)


adult_data = adult_data.dropna()

adult_data["income"] = adult_data["income"].str.replace(".", "", regex=False)

adult_data.to_csv("Real_Datasets/Adult_compressed.csv", index=False)
print(f"预处理完成！Adult数据集共{len(adult_data)}行，保存路径：Real_Datasets/Adult_compressed.csv")
