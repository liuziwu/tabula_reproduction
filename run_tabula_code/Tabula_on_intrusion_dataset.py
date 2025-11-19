import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from tabula import Tabula

# Define column names for the intrusion dataset
intrusion_columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Read raw data
print("Reading raw data...")
train_raw = pd.read_csv(
    "Real_Datasets/Intrusion/kddcup.data_10_percent",
    names=intrusion_columns,
    sep=","
)
test_raw = pd.read_csv(
    "Real_Datasets/Intrusion/kddcup.testdata.unlabeled_10_percent",
    names=intrusion_columns[:-1],  
    sep=","
)

# Print dataset sizes
print(f"Train set (train_raw) actual rows: {len(train_raw)}")
print(f"Test set (test_raw) actual rows: {len(test_raw)}")

# Merge datasets
test_raw["label"] = np.nan  
full_data = pd.concat([train_raw, test_raw], ignore_index=True)
print(f"Total rows after merging: {len(full_data)}")

# Extract labeled data
labeled_data = full_data.dropna(subset=["label"])
print(f"Labeled training data rows: {len(labeled_data)}")

# Split into train and test sets
train_data, test_data = train_test_split(
    labeled_data,
    train_size=40000,  
    test_size=10000,   
    random_state=42,
    stratify=labeled_data["label"]  
)

intrusion_data = pd.concat([train_data, test_data], ignore_index=True)
print(f"Final preprocessed dataset rows: {len(intrusion_data)} (40k+10k)")

# Define categorical columns
categorical_columns = [
    "protocol_type", "service", "flag", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "label"
]

# Convert categorical columns to object type
print("\nConverting categorical columns to object type...")
for col in categorical_columns:
    if col in intrusion_data.columns:
        intrusion_data[col] = intrusion_data[col].astype("object")
        print(f"✅ Column {col} converted to categorical type (object)")
    else:
        print(f"⚠️ Warning: Column {col} not in dataset, skipped")

# Save preprocessed data
output_path = "Real_Datasets/Intrusion_compressed.csv"
intrusion_data.to_csv(output_path, index=False)

# Validation info
print("\n" + "="*60)
print("Intrusion dataset preprocessing completed! Key validation info:")
print(f"1. Final data shape: {intrusion_data.shape} (Expected: (50000, 42))")
print(f"2. Number of categorical columns: {len(intrusion_data.select_dtypes(include=['object']).columns)} (Expected: 21)")
print(f"3. Categorical columns list: {intrusion_data.select_dtypes(include=['object']).columns.tolist()}")
print(f"4. Dataset saved to: {output_path}")
print(f"5. Label distribution (ensure no bias):")
print(intrusion_data["label"].value_counts().head(10))  
print("="*60)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

# Load preprocessed data
data = pd.read_csv("Real_Datasets/Intrusion_compressed.csv")
print("Dataset columns:", data.columns.tolist())

# Ensure categorical columns are object type
for col in categorical_columns:
    data[col] = data[col].astype("object")
    print(f"Column {col} type: {data[col].dtype}")

# Verify categorical columns
new_cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
print("Final number of categorical columns:", len(new_cat_cols))
print("Final categorical columns list:", new_cat_cols)

# Initialize Tabula model
model = Tabula(
    llm='distilgpt2', 
    experiment_dir="intrusion_training", 
    batch_size=32, 
    epochs=1,  
    categorical_columns=categorical_columns
)

# Load pretrained model (uncomment to use)
# model.model.load_state_dict(torch.load("pretrained-model/tabula_pretrained_model.pt"))

# Train the model
model.fit(data)

# Save trained model
torch.save(model.model.state_dict(), "intrusion_training/model_1epoch.pt")

# Generate synthetic data
synthetic_data = model.sample(n_samples=1000, max_length=100)
synthetic_data.to_csv("intrusion_1epoch.csv", index=False)