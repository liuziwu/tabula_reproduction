import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


print("正在读取原始数据...")
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


print(f"训练集（train_raw）实际行数：{len(train_raw)}")
print(f"测试集（test_raw）实际行数：{len(test_raw)}")


test_raw["label"] = np.nan  

full_data = pd.concat([train_raw, test_raw], ignore_index=True)
print(f"合并后总数据行数：{len(full_data)}")

labeled_data = full_data.dropna(subset=["label"])
print(f"有label的训练数据行数：{len(labeled_data)}")


train_data, test_data = train_test_split(
    labeled_data,
    train_size=40000,  
    test_size=10000,   
    random_state=42,
    stratify=labeled_data["label"]  
)

intrusion_data = pd.concat([train_data, test_data], ignore_index=True)
print(f"最终预处理数据集行数：{len(intrusion_data)}（40k+10k）")

categorical_columns = [
    "protocol_type", "service", "flag", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "label"
]

print("\n正在转换分类列数据类型...")
for col in categorical_columns:
    if col in intrusion_data.columns:
        intrusion_data[col] = intrusion_data[col].astype("object")
        print(f"✅ 列 {col} 转换为分类类型（object）")
    else:
        print(f"⚠️  警告：列 {col} 不在数据集中，已跳过")

output_path = "Real_Datasets/Intrusion_compressed.csv"
intrusion_data.to_csv(output_path, index=False)

print("\n" + "="*60)
print("Intrusion数据集预处理完成！关键验证信息：")
print(f"1. 最终数据形状：{intrusion_data.shape}（预期：(50000, 42)）")
print(f"2. 分类列数量：{len(intrusion_data.select_dtypes(include=['object']).columns)}（预期：21）")
print(f"3. 分类列列表：{intrusion_data.select_dtypes(include=['object']).columns.tolist()}")
print(f"4. 数据集保存路径：{output_path}")
print(f"5. 各label分布（确保无偏倚）：")
print(intrusion_data["label"].value_counts().head(10))  
print("="*60)