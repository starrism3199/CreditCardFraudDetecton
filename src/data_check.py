# src/data_check.py
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import datetime
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = {"Time": "float64", "Amount": "float64", "Class": "int64"}
PCA_COLS = [f"V{i}" for i in range(1, 29)]
CLASS_LABEL_MEANING = {0: "正常交易 (Non-Fraud)", 1: "欺诈交易 (Fraud)"}
AMOUNT_MAX_REASONABLE = 100_000
TIME_MAX_REASONABLE = 200_000
IQR_FACTOR = 3.0

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    elif ext in (".parquet",):
        return pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

def generate_report(df: pd.DataFrame, file_path: str, output_path: str) -> str:
    """生成 Markdown 格式的检查报告"""
    md = []
    
    # 1. 基础信息
    md.append("# 数据质量检查报告\n")
    md.append("## 1. 基本信息")
    md.append(f"- **数据来源**: `{file_path}`")
    md.append(f"- **报告生成日期**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **数据集行数**: {df.shape[0]:,} 行")
    md.append(f"- **数据集列数**: {df.shape[1]} 列\n")

    # 2. 字段信息
    md.append("## 2. 字段名称与数据类型")
    md.append("| 字段名 | 数据类型 | 缺失数量 |")
    md.append("| :--- | :--- | :--- |")
    for col in df.columns:
        md.append(f"| {col} | {df[col].dtype} | {df[col].isna().sum()} |")
    md.append("")

    # 3. 类别分布
    md.append("## 3. 样本类别分布 (Class)")
    if "Class" in df.columns:
        total = len(df)
        normal = int((df["Class"] == 0).sum())
        fraud = int((df["Class"] == 1).sum())
        md.append("| 类别定义 | 样本数量 | 占比 |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| 正常交易 (Class=0) | {normal:,} | {normal/total*100:.4f}% |")
        md.append(f"| 欺诈交易 (Class=1) | {fraud:,} | {fraud/total*100:.4f}% |")
    else:
        md.append("- *未找到 Class 字段*")
    md.append("")

    # 4. 缺失与重复
    md.append("## 4. 缺失值与重复值检查")
    total_na = df.isna().sum().sum()
    md.append(f"- **缺失值总数**: {total_na} 个")
    
    dup_count = int(df.duplicated().sum())
    md.append(f"- **完全重复记录数**: {dup_count} 条")
    if dup_count > 0:
        dup_ratio = dup_count / len(df)
        if dup_ratio < 0.01:
            md.append("  - *处理依据*：重复占比 < 1%，影响很小，建议保留并记录，不删除。")
        else:
            md.append("  - *处理依据*：重复占比 > 1%，建议人工抽查后决定是否去重。")
    md.append("")

    # 5. 异常值与数据问题
    md.append("## 5. 需要特别注意的数据问题（异常值）")
    issues = []
    if "Amount" in df.columns:
        neg_amt = int((df["Amount"] < 0).sum())
        zero_amt = int((df["Amount"] == 0).sum())
        if neg_amt > 0: issues.append(f"- **金额为负**: 存在 {neg_amt} 条负金额记录，需确认是否为退款。")
        if zero_amt > 0: issues.append(f"- **金额为零**: 存在 {zero_amt} 条 0 金额记录。")
    if "Time" in df.columns:
        if not df["Time"].is_monotonic_increasing:
            issues.append("- **时间乱序**: Time 字段并非单调递增，需检查采集顺序。")
    
    if issues:
        md.extend(issues)
    else:
        md.append("- 未发现明显的 Amount/Time 异常。")
    md.append("")

    # 6. 预处理建议
    md.append("## 6. 建议采用的预处理方法")
    md.append("根据上述检查，提出以下预处理建议：")
    if "Class" in df.columns:
        fraud_ratio = (df["Class"] == 1).sum() / len(df)
        if fraud_ratio < 0.01:
            md.append("1. **极度不平衡处理**: 欺诈样本极少，建议采用 SMOTE 过采样、欠采样或代价敏感学习。")
    if total_na > 0:
        md.append("2. **缺失值处理**: 对数值型字段使用中位数填充，对类别型字段使用众数填充。")
    else:
        md.append("1. **缺失值处理**: 数据无缺失值，无需处理。")
    if dup_count > 0:
        md.append("3. **重复值处理**: 建议在训练集划分前，根据业务逻辑确认后剔除完全重复行。")
    md.append("4. **特征缩放**: 建议对 `Amount` 和 `Time` 字段使用 `StandardScaler` 或 `RobustScaler` 进行标准化。")

    report_content = "\n".join(md)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    return report_content

def main():
    parser = argparse.ArgumentParser(description="数据检查与报告生成脚本")
    # 设置了默认路径，方便直接运行
    parser.add_argument("--path", default="data/raw/creditcard.csv", help="数据文件路径")
    parser.add_argument("--output", default="results/data_quality.md", help="报告输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[错误] 文件不存在: {args.path}")
        sys.exit(1)

    print(f"正在读取数据: {args.path}")
    df = load_data(args.path)

    print("正在生成报告...")
    report = generate_report(df, args.path, args.output)
    
    print("\n" + "=" * 70)
    print(f"✅ 检查完成！报告已保存至: {args.output}")
    print("=" * 70)
    
    # 顺便在终端也展示一下预览
    print("\n--- 报告预览 ---\n")
    print(report)

if __name__ == "__main__":
    main()