import pandas as pd

# 1. 读取Excel文件并定位“Data”表格
excel_file = pd.ExcelFile("Real_Datasets/Loan/Bank_Personal_Loan_Modelling.xlsx")
data = excel_file.parse("Data")  # 假设表格名为“Data”，需根据实际Excel内容确认

# 2. 处理列名（替换空格为下划线）
data.columns = data.columns.str.replace(" ", "_")

# 3. 确认分类列与连续列（与文档对齐：7个分类列、6个连续列）
categorical_columns = [
    "Family", "Education", "Securities_Account", 
    "CD_Account", "Online", "CreditCard", "Personal_Loan"
]
for col in categorical_columns:
    data[col] = data[col].astype("object")

# 4. 分割训练集/测试集（4k训练、1k测试，分层抽样）
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(
    data,
    train_size=4000,
    test_size=1000,
    random_state=42,
    stratify=data["Personal_Loan"]
)
loan_full = pd.concat([train_data, test_data], ignore_index=True)

# 5. 保存为CSV（供Tabula Notebook调用）
loan_full.to_csv("Real_Datasets/Loan_compressed.csv", index=False)
print("Excel预处理完成，保存路径：Real_Datasets/Loan_compressed.csv")