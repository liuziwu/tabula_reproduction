import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 定义Intrusion数据集完整列名（41个特征列+1个标签列）
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

# 2. 读取原始训练集和测试集（打印行数，确认实际数据规模）
print("正在读取原始数据...")
# 训练集：kddcup.data_10_percent（约49万行）
train_raw = pd.read_csv(
    "Real_Datasets/Intrusion/kddcup.data_10_percent",
    names=intrusion_columns,
    sep=","
)
# 测试集：kddcup.testdata.unlabeled_10_percent（约31万行，无label列）
test_raw = pd.read_csv(
    "Real_Datasets/Intrusion/kddcup.testdata.unlabeled_10_percent",
    names=intrusion_columns[:-1],  # 测试集无label列，排除最后一列
    sep=","
)

# 打印实际行数，便于排查
print(f"训练集（train_raw）实际行数：{len(train_raw)}")
print(f"测试集（test_raw）实际行数：{len(test_raw)}")

# 3. 给测试集补充空label列（后续合并后统一分配，避免长度不匹配）
test_raw["label"] = np.nan  # 用空值占位，合并后一起处理

# 4. 合并训练集和测试集（整体数据约80万行）
full_data = pd.concat([train_raw, test_raw], ignore_index=True)
print(f"合并后总数据行数：{len(full_data)}")

# 5. 筛选有label的数据（仅训练集有原始label，测试集label为空）
labeled_data = full_data.dropna(subset=["label"])
print(f"有label的训练数据行数：{len(labeled_data)}")

# 6. 按文档要求分割为40k训练集 + 10k测试集（从有label的数据中分割，确保label有效）
# 分层抽样：保持label分布一致，符合文档实验要求
train_data, test_data = train_test_split(
    labeled_data,
    train_size=40000,  # 训练集40k行
    test_size=10000,   # 测试集10k行
    random_state=42,
    stratify=labeled_data["label"]  # 关键：按label分层，避免类别偏倚
)

# 合并为最终预处理数据集（供Tabula训练）
intrusion_data = pd.concat([train_data, test_data], ignore_index=True)
print(f"最终预处理数据集行数：{len(intrusion_data)}（40k+10k）")

# 7. 定义分类列列表（20个分类特征列 + 1个label列，符合文档设定）
categorical_columns = [
    "protocol_type", "service", "flag", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "label"
]

# 8. 强制转换分类列数据类型（加入列存在性校验）
print("\n正在转换分类列数据类型...")
for col in categorical_columns:
    if col in intrusion_data.columns:
        intrusion_data[col] = intrusion_data[col].astype("object")
        print(f"✅ 列 {col} 转换为分类类型（object）")
    else:
        print(f"⚠️  警告：列 {col} 不在数据集中，已跳过")

# 9. 保存预处理后的数据集
output_path = "Real_Datasets/Intrusion_compressed.csv"
intrusion_data.to_csv(output_path, index=False)

# 10. 验证预处理结果（关键信息确认）
print("\n" + "="*60)
print("Intrusion数据集预处理完成！关键验证信息：")
print(f"1. 最终数据形状：{intrusion_data.shape}（预期：(50000, 42)）")
print(f"2. 分类列数量：{len(intrusion_data.select_dtypes(include=['object']).columns)}（预期：21）")
print(f"3. 分类列列表：{intrusion_data.select_dtypes(include=['object']).columns.tolist()}")
print(f"4. 数据集保存路径：{output_path}")
print(f"5. 各label分布（确保无偏倚）：")
print(intrusion_data["label"].value_counts().head(10))  # 打印前10个label的数量
print("="*60)