# 数据处理与使用说明 

## 1. 数据概况
原始数据总行数：284,807 行。
正常交易：284,315 行；欺诈交易：492 行。

## 2. 文件说明
数据已按 **70% 训练 / 15% 验证 / 15% 测试** 的比例采用**分层抽样**划分，随机种子为 `42`。
所有处理好的数据均位于 `data/processed/` 目录下：
- `X_train.csv` / `y_train.csv` : 训练集特征和标签
- `X_valid.csv` / `y_valid.csv` : 验证集特征和标签
- `X_test.csv`  / `y_test.csv`  : 测试集特征和标签

*注意：模型训练、调参及尝试 SMOTE 等重采样操作，**只能在训练集 (`X_train`, `y_train`) 上进行**。验证集和测试集绝对不能参与重采样。*

## 3. 特征说明 (防泄漏警告)
- **哪些字段不能作为模型输入**：`Class` 字段是标签（y），已经在 `X_*.csv` 中被剔除，不能作为特征输入模型。
- **数据缩放**：`Time` 和 `Amount` 字段已经使用 `StandardScaler` 进行了标准化。
  - **严禁重新对全量数据进行 `fit` 缩放**。如果你在建模流程中需要用到该缩放器（比如后续有单条新数据预测），请加载保存好的缩放器：`models/scaler.pkl`。
  - 其余 `V1 ~ V28` 字段本身已经是 PCA 降维后的结果，未做二次缩放。

## 4. 快速开始代码示例
```python
import pandas as pd

# 加载训练集
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")['Class']

print(X_train.shape)
```