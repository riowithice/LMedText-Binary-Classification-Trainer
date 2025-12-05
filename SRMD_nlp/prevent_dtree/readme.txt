一、数据准备
1. 数据格式要求
需准备以下两个原始文件：
原始文本文件（默认名称：原始文本.txt）：每行包含一条病历文本数据，编码为 UTF-8
标签文件（默认名称：原始标签.txt）：每行包含一个标签（0 或 1，分别表示 "非预防" 和 "预防"），与文本文件行数严格对应
二、数据处理
使用split_train_dev.py工具将原始数据拆分为训练集和验证集：
bash
运行
# 基本用法（默认按8:2比例拆分）
python split_train_dev.py --txt 原始文本.txt --lbl 原始标签.txt

# 自定义拆分比例（如7:3）
python split_train_dev.py --txt 原始文本.txt --lbl 原始标签.txt --ratio 0.7
执行后会生成 4 个文件：
train.txt：训练集文本
train_label.txt：训练集标签
dev.txt：验证集文本
dev_label.txt：验证集标签
三、模型训练
1. 基础版本训练
bash
运行
python prevent_dtree.py --data_dir ./数据目录 --yml prevent_rules.yml
--data_dir：存放训练集和验证集的文件夹路径（需包含上述 4 个拆分后的文件）
--yml：规则配置文件（定义特征工程规则）
输出：生成模型文件prevent_dtree.joblib
2. SMOTE 改进版本训练（处理类别类别不平衡）
bash
运行
python prevent_dtree_smote.py --data_dir ./数据目录 --yml prevent_rules.yml
功能：在基础版本上增加 SMOTE 过采样 + 随机欠采样处理
输出：生成模型文件prevent_dtree_smote.joblib及plots文件夹（内含 6 种评估图表）
四、结果可视化与评估
以下脚本需在模型训练完成后执行，基于训练好的prevent_dtree.joblib生成可视化结果：
脚本名称	功能	输出文件
tree_vis.py	生成决策树结构图（中英文版本）	tree_english.png、tree_chinese.png
confusion_matrix_heatmap.py	生成混淆矩阵热力图（中英文版本）	confusion_matrix_cn.png、confusion_matrix_en.png
feat_importance.py	生成 Top30 特征重要性条形图（中英文中英文版本）	feat_importance_cn.png、feat_importance_en.png
pr_roc_curve.py	生成 PR 曲线和 ROC 曲线（含 AUC 值）	pr_curve.png、roc_curve.png
calibration_curve.py	生成模型校准曲线	calibration.png
threshold_scan.py	生成不同阈值下的精确率、召回率和 F1 曲线	threshold_scan.png
执行示例
bash
运行
# 生成决策树可视化
python tree_vis.py

# 生成特征重要性图
python feat_importance.py

# 生成阈值扫描图
python threshold_scan.py