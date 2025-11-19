import os
import pandas as pd
import torch
from tabula import Tabula 
from tabula.tabula_start import RandomStart

# Specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Preprocess raw data
col_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]

data_raw = pd.read_csv("Real_Datasets/Covertype/covtype.data", names=col_names)

categorical_columns = [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]
for col in categorical_columns:
    data_raw[col] = data_raw[col].astype("object")

data_raw.to_csv("Real_Datasets/Covertype_compressed.csv", index=False)
print(f"Preprocessing completed! Covertype dataset has {len(data_raw)} rows, saved to: Real_Datasets/Covertype_compressed.csv")


# Load compressed dataset
data = pd.read_csv("Real_Datasets/Covertype_compressed.csv")
all_columns = data.columns.tolist()
print(f"Compressed dataset columns: {all_columns}")
print(f"Original rows in compressed dataset: {len(data)}")


# Process categorical columns
existing_cat_cols = [col for col in categorical_columns if col in data.columns]
print(f"Existing categorical columns: {existing_cat_cols}, total: {len(existing_cat_cols)}")

for col in existing_cat_cols:
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].astype("object", errors="ignore")
print(f"Valid rows after processing: {len(data)}")


# Initialize model
model = Tabula(
    llm='distilgpt2', 
    experiment_dir="covertype_training", 
    batch_size=32, 
    epochs=30, 
    categorical_columns=existing_cat_cols
)


# Load pretrained model
model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"),
    strict=False
)


# Train model
model.fit(data)


# Save trained model
torch.save(model.model.state_dict(), "covertype_training/model_30epoch.pt")


# Generate synthetic data
start_generator = RandomStart(
    all_columns=all_columns,
    tokenizer=model.tokenizer
)
synthetic_data = model.sample(
    n_samples=50000, 
    max_length=200,
    start_col_generator=start_generator
)
synthetic_data.to_csv("covertype_30epoch.csv", index=False)

print("Code execution completed, synthetic data saved")