# train_smote_catboost_shap.py
import os
import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

EXCEL_FILE = 'output.xlsx'
MODEL_DIR = 'model_smote_catboost'
FIG_DIR = 'figures_smote_catboost'  # 图片保存目录
BLACKLIST_CONFIG = 'blacklist_config.txt'  # 屏蔽词配置文件
TRANSLATION_DICT_FILE = 'cn2en_dict.txt'  # 中英对照字典文件

for d in [MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 字典加载函数 ==========
def load_translation_dict(filename=TRANSLATION_DICT_FILE):
    """从文本文件加载中英对照字典"""
    cn2en = {}
    
    print(f"\n{'='*60}")
    print(f"加载中英对照字典文件: {filename}")
    print('='*60)
    
    if not os.path.exists(filename):
        print(f"⚠️  警告: 字典文件 {filename} 不存在，使用空字典")
        print("请创建此文件，格式：每行'中文=英文'")
        return cn2en
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        cn = parts[0].strip()
                        en = parts[1].strip()
                        if cn and en:  # 确保键值都不为空
                            cn2en[cn] = en
        
        print(f"✓ 字典加载完成: {len(cn2en)} 个词条")
        
        # 显示部分词条
        if cn2en:
            print("\n字典词条示例:")
            items = list(cn2en.items())
            for i, (cn, en) in enumerate(items[:10]):
                print(f"  {i+1:2d}. {cn:<15} -> {en}")
            if len(items) > 10:
                print(f"  ... 还有 {len(items)-10} 个词条")
        
        return cn2en
        
    except Exception as e:
        print(f"❌ 加载字典文件时出错: {e}")
        print("使用空字典继续运行...")
        return {}

# ========== 屏蔽词管理函数 ==========
def load_blacklist_config(filename=BLACKLIST_CONFIG):
    """从配置文件加载屏蔽词，如果文件不存在则报错"""
    exact_blacklist = []      # 精确匹配的屏蔽词
    pattern_blacklist = []    # 正则表达式模式
    
    print(f"\n{'='*60}")
    print("加载屏蔽词配置文件")
    print('='*60)
    
    if not os.path.exists(filename):
        print(f"❌ 错误: 屏蔽词配置文件 {filename} 不存在！")
        print("请创建此文件并添加需要屏蔽的词汇。")
        print("文件格式：每行一个屏蔽词，支持通配符*和正则表达式")
        print("示例：")
        print("  签名*        # 屏蔽所有以'签名'开头的特征")
        print("  陈梅琴       # 精确屏蔽'陈梅琴'")
        print("  ^\\d+$       # 屏蔽纯数字")
        raise FileNotFoundError(f"屏蔽词配置文件 {filename} 不存在")
    
    print(f"正在读取配置文件: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '*' in line:
                    # 处理通配符模式
                    pattern = line.replace('*', '.*')
                    pattern_blacklist.append({
                        'original': line,
                        'pattern': pattern
                    })
                    print(f"  行{line_num}: 模式屏蔽 - {line}")
                elif line.startswith('^') or '\\' in line or '[' in line or '(' in line:
                    # 正则表达式模式
                    pattern_blacklist.append({
                        'original': line,
                        'pattern': line
                    })
                    print(f"  行{line_num}: 正则屏蔽 - {line}")
                else:
                    # 精确匹配词
                    exact_blacklist.append({
                        'word': line,
                        'line': line_num
                    })
                    print(f"  行{line_num}: 精确屏蔽 - {line}")
    
    print(f"\n✓ 加载完成:")
    print(f"  精确屏蔽词: {len(exact_blacklist)} 个")
    print(f"  模式屏蔽词: {len(pattern_blacklist)} 个")
    
    if exact_blacklist:
        print("\n精确屏蔽词示例:")
        for i, item in enumerate(exact_blacklist[:10]):
            print(f"  {i+1:2d}. {item['word']}")
        if len(exact_blacklist) > 10:
            print(f"  ... 还有 {len(exact_blacklist)-10} 个")
    
    if pattern_blacklist:
        print("\n模式屏蔽词示例:")
        for i, item in enumerate(pattern_blacklist[:10]):
            print(f"  {i+1:2d}. {item['original']}")
        if len(pattern_blacklist) > 10:
            print(f"  ... 还有 {len(pattern_blacklist)-10} 个")
    
    return exact_blacklist, pattern_blacklist

def is_feature_blacklisted(feature_name, exact_blacklist, pattern_blacklist):
    """检查特征是否被屏蔽"""
    if not isinstance(feature_name, str):
        feature_name = str(feature_name)
    
    feature_name = feature_name.strip()
    
    # 1. 检查精确匹配
    for item in exact_blacklist:
        if item['word'] == feature_name:
            return True, f"精确匹配: {item['word']}"
    
    # 2. 检查模式匹配
    for item in pattern_blacklist:
        try:
            if re.search(item['pattern'], feature_name):
                return True, f"模式匹配: {item['original']}"
        except re.error as e:
            # 如果正则表达式有误，尝试简单的字符串匹配
            print(f"⚠️  正则表达式错误: {item['pattern']} - {e}")
    
    return False, None

def filter_features(feature_names, feature_importances, exact_blacklist, pattern_blacklist):
    """过滤特征，返回保留的特征索引和名称"""
    keep_indices = []
    keep_names = []
    keep_importances = []
    filtered_info = []  # 被过滤的特征信息
    
    for idx, feat_name in enumerate(feature_names):
        is_blacklisted, reason = is_feature_blacklisted(feat_name, exact_blacklist, pattern_blacklist)
        
        if is_blacklisted:
            filtered_info.append({
                'feature': feat_name,
                'importance': feature_importances[idx] if idx < len(feature_importances) else 0,
                'reason': reason
            })
        else:
            keep_indices.append(idx)
            keep_names.append(feat_name)
            if idx < len(feature_importances):
                keep_importances.append(feature_importances[idx])
            else:
                keep_importances.append(0)
    
    # 按重要性排序被过滤的特征
    filtered_info.sort(key=lambda x: x['importance'], reverse=True)
    
    return keep_indices, keep_names, keep_importances, filtered_info

# ========== 翻译管理函数 ==========
def translate_feature(cn_feat, cn2en_dict):
    """翻译中文特征"""
    if cn_feat in cn2en_dict:
        return cn2en_dict[cn_feat]
    
    # 如果是n-gram特征，尝试分词翻译
    if ' ' in cn_feat:
        parts = cn_feat.split(' ')
        translated_parts = []
        for part in parts:
            if part in cn2en_dict:
                translated_parts.append(cn2en_dict[part])
            else:
                translated_parts.append(part)
        return ' '.join(translated_parts)
    
    # 如果找不到翻译，返回原词
    return cn_feat

def print_translation_table(top_cn, top_en, importances, top_idx, save_path=None, top_n=10):
    """打印中英文对照表"""
    print(f"\n{'='*60}")
    print(f"SHAP图 Y轴标签 - 中英文对照表 (Top-{top_n})")
    print('='*60)
    print(f"{'排名':<5} {'中文特征':<40} {'英文翻译':<40} {'重要性':<10}")
    print('-'*100)
    
    for i, (cn, en, idx) in enumerate(zip(top_cn, top_en, top_idx), 1):
        cn_display = cn[:38] + "..." if len(cn) > 40 else cn.ljust(40)
        en_display = en[:38] + "..." if len(en) > 40 else en.ljust(40)
        importance_val = importances[idx] if idx < len(importances) else 0
        print(f"{i:<5} {cn_display:<40} {en_display:<40} {importance_val:<10.6f}")
    
    # 保存对照表到文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"SHAP图 Y轴标签 - 中英文对照表 (Top-{top_n})\n")
            f.write('='*60 + '\n')
            f.write(f"{'排名':<5} {'中文特征':<40} {'英文翻译':<40} {'重要性':<10}\n")
            f.write('-'*100 + '\n')
            for i, (cn, en, idx) in enumerate(zip(top_cn, top_en, top_idx), 1):
                importance_val = importances[idx] if idx < len(importances) else 0
                f.write(f"{i:<5} {cn:<40} {en:<40} {importance_val:<10.6f}\n")
        print(f"\n✓ 对照表已保存到: {save_path}")

# ========== SHAP标签输出函数 ==========
def save_shap_labels(top_cn, top_en, shap_values, save_dir=FIG_DIR):
    """
    保存SHAP图的纵轴标签信息
    
    参数:
    - top_cn: 中文特征名称列表
    - top_en: 英文特征名称列表
    - shap_values: SHAP值数组
    - save_dir: 保存目录
    """
    
    # 计算每个特征的平均SHAP绝对值（重要性）
    if len(shap_values.shape) == 2:
        feature_importance = np.abs(shap_values).mean(axis=0)
    else:
        feature_importance = np.abs(shap_values)
    
    # 确保长度一致
    min_len = min(len(top_cn), len(top_en), len(feature_importance))
    top_cn = top_cn[:min_len]
    top_en = top_en[:min_len]
    feature_importance = feature_importance[:min_len]
    
    # 保存中文标签文件
    cn_file = os.path.join(save_dir, 'shap_yaxis_labels_cn.txt')
    with open(cn_file, 'w', encoding='utf-8') as f:
        f.write("SHAP图纵轴标签 - 中文版 (从上到下的顺序)\n")
        f.write("="*60 + "\n\n")
        f.write("注意: 这是SHAP图中从顶部到底部的特征顺序\n")
        f.write("在图中，最重要的特征显示在最顶部\n\n")
        f.write(f"{'排名':<5} {'中文特征':<50} {'平均SHAP绝对值':<15}\n")
        f.write('-'*70 + '\n')
        
        for i, (cn, importance) in enumerate(zip(reversed(top_cn), reversed(feature_importance)), 1):
            f.write(f"{i:<5} {cn:<50} {importance:<15.6f}\n")
    
    print(f"✓ 中文纵轴标签已保存到: {cn_file}")
    
    # 保存英文标签文件
    en_file = os.path.join(save_dir, 'shap_yaxis_labels_en.txt')
    with open(en_file, 'w', encoding='utf-8') as f:
        f.write("SHAP图纵轴标签 - 英文版 (从上到下的顺序)\n")
        f.write("="*60 + "\n\n")
        f.write("Note: This shows the feature order from top to bottom in SHAP plots\n")
        f.write("In the plots, the most important features appear at the top\n\n")
        f.write(f"{'Rank':<5} {'English Feature':<50} {'Mean |SHAP|':<15}\n")
        f.write('-'*70 + '\n')
        
        for i, (en, importance) in enumerate(zip(reversed(top_en), reversed(feature_importance)), 1):
            f.write(f"{i:<5} {en:<50} {importance:<15.6f}\n")
    
    print(f"✓ 英文纵轴标签已保存到: {en_file}")
    
    # 保存中英文对照文件（按图中顺序）
    bilingual_file = os.path.join(save_dir, 'shap_yaxis_labels_bilingual.txt')
    with open(bilingual_file, 'w', encoding='utf-8') as f:
        f.write("SHAP图纵轴标签 - 中英文对照 (从上到下的顺序)\n")
        f.write("="*80 + "\n\n")
        f.write("注意: 这是SHAP图中从顶部到底部的特征顺序\n")
        f.write("在图中，最重要的特征显示在最顶部\n\n")
        f.write(f"{'排名':<5} {'中文特征':<40} {'英文翻译':<40} {'平均SHAP绝对值':<15}\n")
        f.write('-'*100 + '\n')
        
        for i, (cn, en, importance) in enumerate(zip(reversed(top_cn), reversed(top_en), reversed(feature_importance)), 1):
            f.write(f"{i:<5} {cn:<40} {en:<40} {importance:<15.6f}\n")
    
    print(f"✓ 中英文对照纵轴标签已保存到: {bilingual_file}")
    
    # 打印标签信息
    print(f"\n{'='*60}")
    print("SHAP图纵轴标签信息")
    print('='*60)
    print(f"总特征数: {len(top_cn)}")
    print("\nSHAP图中的特征顺序 (从上到下):")
    print("-" * 80)
    for i, (cn, en, importance) in enumerate(zip(reversed(top_cn), reversed(top_en), reversed(feature_importance)), 1):
        cn_display = cn[:35] + "..." if len(cn) > 35 else cn.ljust(35)
        en_display = en[:35] + "..." if len(en) > 35 else en.ljust(35)
        print(f"{i:2d}. {cn_display:<35} ({en_display:<35}) |SHAP|={importance:.6f}")

# ========== 主要处理流程 ==========
def main():
    print("="*60)
    print("SMOTE + CatBoost 文本二分类分析")
    print("="*60)
    
    # 1. 加载中英对照字典
    cn2en = load_translation_dict()
    
    # 2. 加载屏蔽词配置
    exact_blacklist, pattern_blacklist = load_blacklist_config()
    
    # 3. 读数据 + 分词
    def read_excel(path):
        df = pd.read_excel(path, engine='openpyxl').iloc[:, 1:3]
        df.columns = ['text', 'label']
        return df.dropna(subset=['text', 'label'])
    
    def cut(txt):
        return ' '.join(jieba.lcut(txt))
    
    df = read_excel(EXCEL_FILE)
    df['text_cut'] = df['text'].astype(str).apply(cut)
    
    # 4. 中文停用词表
    with open('cn_stop.txt', encoding='utf-8') as f:
        stop = list(set(f.read().split()))
    
    # 5. 向量化
    tfidf = TfidfVectorizer(lowercase=False, stop_words=stop, ngram_range=(1, 2), max_df=0.9, min_df=2)
    X = tfidf.fit_transform(df['text_cut'])
    y = df['label'].astype(int)
    
    print(f"\n特征维度: {X.shape[1]}")
    print(f"类别分布: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # 6. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 7. 应用SMOTE仅在训练集上
    print("\n应用SMOTE平衡训练数据...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"SMOTE后训练数据: 0={sum(y_train_bal==0)}, 1={sum(y_train_bal==1)}")
    
    # 8. 训练CatBoost模型
    clf = CatBoostClassifier(
        random_state=42,
        verbose=0,
        iterations=500,  # 迭代次数
        learning_rate=0.05,  # 学习率
        depth=6,  # 树深度
        l2_leaf_reg=3,  # L2正则化
        border_count=32,  # 特征分箱数
        loss_function='Logloss',  # 损失函数
        eval_metric='F1',  # 评估指标
        early_stopping_rounds=50  # 早停轮数
    )
    
    # 转换为密集矩阵供CatBoost使用
    X_train_bal_dense = X_train_bal.toarray()
    clf.fit(X_train_bal_dense, y_train_bal)
    
    # 9. 预测概率
    X_test_dense = X_test.toarray()
    y_prob = clf.predict_proba(X_test_dense)[:, 1]
    
    # 10. 自动寻找最佳阈值（以 F1 分数为目标）
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']  # 获取类别 1 的 F1 分数
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f'最佳阈值: {best_threshold:.2f}, 最佳 F1 分数: {best_f1:.4f}')
    
    # 11. 使用最佳阈值进行预测
    y_pred_best = (y_prob >= best_threshold).astype(int)
    
    # 12. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix (SMOTE+CatBoost, Thr={best_threshold:.2f})')
    plt.savefig(os.path.join(FIG_DIR, 'cm_smote_catboost.png'), dpi=300)
    plt.close()
    print("✓ 混淆矩阵保存成功")
    
    # 13. ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC – SMOTE+CatBoost')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIG_DIR, 'roc_smote_catboost.png'), dpi=300)
    plt.close()
    print("✓ ROC曲线保存成功")
    
    # 14. PR 曲线
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    average_precision = auc(recall, precision)
    
    plt.figure(figsize=(5, 5))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(label=f'AP = {average_precision:.3f}')
    plt.plot([0, 1], [np.mean(y_test), np.mean(y_test)], '--', color='gray', label=f'Baseline = {np.mean(y_test):.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR – SMOTE+CatBoost')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIG_DIR, 'pr_smote_catboost.png'), dpi=300)
    plt.close()
    print("✓ PR曲线保存成功")
    
    # 15. 评估报告
    print('\n===== SMOTE+CatBoost 最佳阈值评估 =====')
    print(confusion_matrix(y_test, y_pred_best))
    print(classification_report(y_test, y_pred_best, digits=4))
    
    # 16. SHAP分析（树模型使用TreeExplainer）
    print(f"\n{'='*60}")
    print("开始SHAP分析（带屏蔽词过滤）")
    print('='*60)
    
    # 16.1 采样用于SHAP分析
    sample_size = min(200, X_test.shape[0])
    test_idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_test_sample_dense = X_test_dense[test_idx]
    
    # 16.2 使用TreeExplainer（适合树模型）
    print("使用TreeExplainer计算SHAP值...")
    
    try:
        # 对于CatBoost等树模型，我们使用TreeExplainer
        explainer = shap.TreeExplainer(clf)
        print("✓ 使用TreeExplainer")
    except Exception as e:
        print(f"❌ SHAP初始化失败: {e}")
        print("跳过SHAP分析...")
        explainer = None
    
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X_test_sample_dense)
            print("✓ SHAP值计算成功")
            
            # 对于TreeExplainer，可能需要处理输出格式
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1]  # 正类
            else:
                shap_matrix = shap_values
            
            print(f"SHAP矩阵形状: {shap_matrix.shape}")
            
            # 16.3 获取特征名
            feature_names_list = tfidf.get_feature_names_out().tolist()
            
            # 16.4 计算特征重要性（基于SHAP绝对值的平均值）
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            
            # 16.5 使用屏蔽词过滤特征
            keep_indices, keep_names, keep_shap, filtered_info = filter_features(
                feature_names_list, mean_abs_shap, exact_blacklist, pattern_blacklist)
            
            print(f"\n原始特征数: {len(feature_names_list)}")
            print(f"保留特征数: {len(keep_indices)}")
            print(f"过滤特征数: {len(filtered_info)}")
            
            # 显示被过滤的重要特征
            if filtered_info:
                print(f"\n被过滤的重要特征 (前10个):")
                print("-" * 80)
                for i, item in enumerate(filtered_info[:10]):
                    feat_display = item['feature'][:60] + "..." if len(item['feature']) > 60 else item['feature'].ljust(60)
                    print(f"{i+1:2d}. {feat_display:<60} 重要性: {item['importance']:.6f} ({item['reason']})")
            
            # 16.6 从过滤后的特征中选择Top-10
            TOP_N = 10
            
            if len(keep_shap) >= TOP_N:
                # 对保留的特征按SHAP值排序
                keep_shap_array = np.array(keep_shap)
                top_filtered_idx = np.argsort(keep_shap_array)[::-1][:TOP_N]
                top_idx = [keep_indices[i] for i in top_filtered_idx]
                print(f"\n✓ 从过滤后的特征中选择了Top-{TOP_N}")
            else:
                print(f"\n⚠️  过滤后特征不足{TOP_N}个({len(keep_shap)})，使用原始Top特征")
                original_top_idx = np.argsort(mean_abs_shap)[::-1][:TOP_N]
                top_idx = original_top_idx.tolist()
            
            # 16.7 准备Top-10特征
            top_cn = [feature_names_list[i] for i in top_idx]
            top_shap = shap_matrix[:, top_idx]
            top_features = X_test_sample_dense[:, top_idx]
            
            # 16.8 翻译特征
            top_en = [translate_feature(feat, cn2en) for feat in top_cn]
            
            # 16.9 输出中英文对照表
            print_translation_table(
                top_cn, top_en, mean_abs_shap, top_idx,
                os.path.join(FIG_DIR, 'shap_translation_table_smote_catboost.txt'),
                top_n=TOP_N
            )
            
            # 16.10 保存SHAP图纵轴标签
            print(f"\n{'='*60}")
            print("保存SHAP图纵轴标签")
            print('='*60)
            save_shap_labels(top_cn, top_en, top_shap, FIG_DIR)
            
            # 17. 绘制SHAP图
            # 17.1 Beeswarm图（英文）
            try:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(top_shap, top_features,
                                  feature_names=top_en,
                                  plot_type="dot", show=False,
                                  max_display=TOP_N)
                plt.title(f"Top-{TOP_N} SHAP Feature Importance (Beeswarm)", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"shap_beeswarm_smote_catboost_top{TOP_N}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✓ Beeswarm图(英文)保存成功 (Top-{TOP_N})")
            except Exception as e:
                print(f"绘制beeswarm图时出错: {e}")
            
            # 17.2 Bar图（英文）
            try:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(top_shap, top_features,
                                  feature_names=top_en,
                                  plot_type="bar", show=False,
                                  max_display=TOP_N)
                plt.title(f"Top-{TOP_N} SHAP Feature Importance (Bar Chart)", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"shap_bar_smote_catboost_top{TOP_N}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✓ Bar图(英文)保存成功 (Top-{TOP_N})")
            except Exception as e:
                print(f"绘制bar图时出错: {e}")
            
            # 17.3 可选：绘制中文标签的图
            try:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(top_shap, top_features,
                                  feature_names=top_cn,
                                  plot_type="dot", show=False,
                                  max_display=TOP_N)
                plt.title(f"Top-{TOP_N} SHAP 特征重要性 (Beeswarm - 中文)", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"shap_beeswarm_cn_smote_catboost_top{TOP_N}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✓ Beeswarm图(中文)保存成功 (Top-{TOP_N})")
            except Exception as e:
                print(f"绘制中文beeswarm图时出错: {e}")
            
            # 17.4 可选：绘制中文标签的Bar图
            try:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(top_shap, top_features,
                                  feature_names=top_cn,
                                  plot_type="bar", show=False,
                                  max_display=TOP_N)
                plt.title(f"Top-{TOP_N} SHAP 特征重要性 (Bar Chart - 中文)", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"shap_bar_cn_smote_catboost_top{TOP_N}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✓ Bar图(中文)保存成功 (Top-{TOP_N})")
            except Exception as e:
                print(f"绘制中文bar图时出错: {e}")
                
        except Exception as e:
            print(f"❌ 计算SHAP值时出错: {e}")
            print("跳过SHAP分析...")
    
    # 18. 特征重要性（CatBoost内置特征重要性）
    print(f"\n{'='*60}")
    print("CatBoost 特征重要性分析")
    print('='*60)
    
    # 获取CatBoost内置特征重要性
    feature_importance = clf.get_feature_importance()
    
    # 获取特征名
    feature_names_list = tfidf.get_feature_names_out().tolist()
    
    # 过滤和排序特征重要性
    keep_importance_indices, keep_importance_names, keep_importance_values, _ = filter_features(
        feature_names_list, feature_importance, exact_blacklist, pattern_blacklist)
    
    TOP_N = 10
    if len(keep_importance_values) >= TOP_N:
        keep_importance_array = np.array(keep_importance_values)
        top_importance_idx = np.argsort(keep_importance_array)[::-1][:TOP_N]
        top_importance_indices = [keep_importance_indices[i] for i in top_importance_idx]
    else:
        original_top_idx = np.argsort(feature_importance)[::-1][:TOP_N]
        top_importance_indices = original_top_idx.tolist()
    
    # 输出Top特征重要性
    print(f"\nTop-{TOP_N} 特征重要性 (CatBoost):")
    print("-" * 80)
    for i, idx in enumerate(top_importance_indices, 1):
        feat_name = feature_names_list[idx]
        feat_en = translate_feature(feat_name, cn2en)
        importance_val = feature_importance[idx]
        feat_display = feat_name[:40] + "..." if len(feat_name) > 40 else feat_name.ljust(40)
        en_display = feat_en[:30] + "..." if len(feat_en) > 30 else feat_en.ljust(30)
        print(f"{i:2d}. {feat_display:<40} ({en_display:<30}) 重要性: {importance_val:+.6f}")
    
    # 19. 保存模型
    joblib.dump(clf, os.path.join(MODEL_DIR, 'catboost_smote.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_smote.pkl'))
    print(f"\n✓ 模型已保存到 {MODEL_DIR}")
    
    print(f"\n{'='*60}")
    print("SMOTE + CatBoost 分析完成！")
    print(f"所有图表已保存到: {FIG_DIR}/")
    print(f"特征分析使用Top-{TOP_N}特征")
    print('='*60)

if __name__ == "__main__":
    main()