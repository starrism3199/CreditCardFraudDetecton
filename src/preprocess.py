# src/preprocess.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 路径配置
DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"  # 用于保存缩放器对象

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("1. 正在读取原始数据...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. 分离特征 X 和标签 y
    print("2. 分离特征与标签...")
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 3. 按统一规则划分训练集(70%)、验证集(15%)、测试集(15%)
    # 注意：使用 stratify=y 进行分层抽样，保证三个子集的欺诈比例一致
    print("3. 正在进行分层抽样划分 (70/15/15)...")
    # 先划出 70% 训练集，剩下的 30% 作为临时集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # 再把临时集均分为 15% 验证集 和 15% 测试集
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # 4. 特征缩放 (严格执行防止数据泄漏)
    # 项目要求处理 Time 和 Amount。由于 V1~V28 已经是 PCA 结果，通常不需要再缩放
    print("4. 正在拟合 StandardScaler (仅在训练集上 fit)...")
    scaler = StandardScaler()
    
    # 只对 Time 和 Amount 缩放
    cols_to_scale = ['Time', 'Amount']
    
    # 【核心规则】：只在训练集上 fit_transform
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    # 【核心规则】：验证集和测试集只能 transform，绝对不能 fit
    X_valid[cols_to_scale] = scaler.transform(X_valid[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # 5. 保存缩放器对象
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   - 缩放器已保存至: {scaler_path}")
    
    # 6. 输出统一数据到 CSV
    print("5. 正在保存划分后的数据...")
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    
    X_valid.to_csv(f"{OUTPUT_DIR}/X_valid.csv", index=False)
    y_valid.to_csv(f"{OUTPUT_DIR}/y_valid.csv", index=False)
    
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)
    
    # 7. 生成数据使用说明 (README.md)
    print("6. 正在生成数据使用说明书...")
    
    # 使用列表拼接，避免嵌套 Markdown 代码块导致的语法错误
    readme_lines = []
    readme_lines.append("# 数据处理与使用说明 \n")
    readme_lines.append("## 1. 数据概况")
    readme_lines.append(f"原始数据总行数：{len(df):,} 行。")
    readme_lines.append(f"正常交易：{len(df[df['Class'] == 0]):,} 行；欺诈交易：{len(df[df['Class'] == 1]):,} 行。\n")
    
    readme_lines.append("## 2. 文件说明")
    readme_lines.append("数据已按 **70% 训练 / 15% 验证 / 15% 测试** 的比例采用**分层抽样**划分，随机种子为 `42`。")
    readme_lines.append("所有处理好的数据均位于 `data/processed/` 目录下：")
    readme_lines.append("- `X_train.csv` / `y_train.csv` : 训练集特征和标签")
    readme_lines.append("- `X_valid.csv` / `y_valid.csv` : 验证集特征和标签")
    readme_lines.append("- `X_test.csv`  / `y_test.csv`  : 测试集特征和标签\n")
    readme_lines.append("*注意：模型训练、调参及尝试 SMOTE 等重采样操作，**只能在训练集 (`X_train`, `y_train`) 上进行**。验证集和测试集绝对不能参与重采样。*\n")
    
    readme_lines.append("## 3. 特征说明 (防泄漏警告)")
    readme_lines.append("- **哪些字段不能作为模型输入**：`Class` 字段是标签（y），已经在 `X_*.csv` 中被剔除，不能作为特征输入模型。")
    readme_lines.append("- **数据缩放**：`Time` 和 `Amount` 字段已经使用 `StandardScaler` 进行了标准化。")
    readme_lines.append("  - **严禁重新对全量数据进行 `fit` 缩放**。如果你在建模流程中需要用到该缩放器（比如后续有单条新数据预测），请加载保存好的缩放器：`models/scaler.pkl`。")
    readme_lines.append("  - 其余 `V1 ~ V28` 字段本身已经是 PCA 降维后的结果，未做二次缩放。\n")
    
    readme_lines.append("## 4. 快速开始代码示例")
    readme_lines.append("```python")
    readme_lines.append("import pandas as pd")
    readme_lines.append("")
    readme_lines.append("# 加载训练集")
    readme_lines.append('X_train = pd.read_csv("data/processed/X_train.csv")')
    readme_lines.append('y_train = pd.read_csv("data/processed/y_train.csv")[\'Class\']')
    readme_lines.append("")
    readme_lines.append("print(X_train.shape)")
    readme_lines.append("```")
    
    # 拼接并写入文件
    readme_content = "\n".join(readme_lines)
    with open(f"{OUTPUT_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ 预处理完成！")
    print(f"   训练集: {X_train.shape[0]} 条")
    print(f"   验证集: {X_valid.shape[0]} 条")
    print(f"   测试集: {X_test.shape[0]} 条")
    print(f"   数据已保存至: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()