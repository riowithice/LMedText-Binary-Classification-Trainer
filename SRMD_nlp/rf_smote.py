# -*- coding: utf-8 -*-
"""
SMOTE + RandomForest 文本二分类
一键产出 6 张图 → figs3/
SHAP 可解释性（带屏蔽词过滤功能）
"""
import os, jieba, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
import joblib
from imblearn.over_sampling import SMOTE
import shap
import re

# ========== 路径和配置 ==========
EXCEL_FILE = 'output.xlsx'
MODEL_DIR  = 'model'
FIG_DIR    = 'figs3'
BLACKLIST_CONFIG = 'blacklist_config.txt'  # 屏蔽词配置文件

for d in [MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 中英对照字典 ==========
cn2en = {
    '骨折': 'fracture', '入院': 'admission', '辅助 检查': 'auxiliary examination',
    '患者': 'patient', '辅助': 'auxiliary', '左侧': 'left side', '检查': 'examination',
    '收住': 'hospitalization', '小时': 'hour', '外伤': 'trauma', '收住 入院': 'admission to hospital',
    '医师': 'physician', '初步': 'preliminary', 'CT': 'CT', '损伤': 'injury',
    '初步 诊断': 'preliminary diagnosis', '疼痛': 'pain', '诊断': 'diagnosis', '出血': 'bleeding',
    '门诊': 'outpatient', '治疗': 'treatment', '少量': 'small amount', '考虑': 'consideration',
    '目前': 'currently', '提示': 'indication', '小时 入院': 'hour admission', '记录': 'record',
    '血小板': 'platelet', '明显': 'obvious', '活动': 'activity','主动脉 冠脉': 'aorta coronary artery',
    '临床 胆囊结石': 'clinical gallbladder stones',
    '增大 胰头': 'enlarged pancreatic head',
    '骨折 头皮': 'fracture scalp',
    '出血 气管': 'bleeding trachea',
    '多发 钙化': 'multiple calcifications',
    '不良反应': 'adverse reaction',
    '髋关节': 'hip joint',
    '心脏 彩超': 'cardiac color ultrasound',
    '腹痛': 'abdominal pain'
}

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

def filter_features(feature_names, shap_values, exact_blacklist, pattern_blacklist):
    """过滤特征，返回保留的特征索引和名称"""
    keep_indices = []
    keep_names = []
    keep_shap = []
    filtered_info = []  # 被过滤的特征信息
    
    for idx, feat_name in enumerate(feature_names):
        is_blacklisted, reason = is_feature_blacklisted(feat_name, exact_blacklist, pattern_blacklist)
        
        if is_blacklisted:
            filtered_info.append({
                'feature': feat_name,
                'shap': shap_values[idx] if idx < len(shap_values) else 0,
                'reason': reason
            })
        else:
            keep_indices.append(idx)
            keep_names.append(feat_name)
            if idx < len(shap_values):
                keep_shap.append(shap_values[idx])
            else:
                keep_shap.append(0)
    
    # 按SHAP值排序被过滤的特征
    filtered_info.sort(key=lambda x: x['shap'], reverse=True)
    
    return keep_indices, keep_names, keep_shap, filtered_info

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
    
    return cn_feat

def print_translation_table(top_cn, top_en, shap_values, top_idx, save_path=None, top_n=10):
    """打印中英文对照表"""
    print(f"\n{'='*60}")
    print(f"SHAP图 Y轴标签 - 中英文对照表 (Top-{top_n})")
    print('='*60)
    print(f"{'排名':<5} {'中文特征':<40} {'英文翻译':<40} {'SHAP值':<10}")
    print('-'*100)
    
    for i, (cn, en, idx) in enumerate(zip(top_cn, top_en, top_idx), 1):
        cn_display = cn[:38] + "..." if len(cn) > 40 else cn.ljust(40)
        en_display = en[:38] + "..." if len(en) > 40 else en.ljust(40)
        shap_val = shap_values[idx] if idx < len(shap_values) else 0
        print(f"{i:<5} {cn_display:<40} {en_display:<40} {shap_val:<10.6f}")
    
    # 保存对照表到文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"SHAP图 Y轴标签 - 中英文对照表 (Top-{top_n})\n")
            f.write('='*60 + '\n')
            f.write(f"{'排名':<5} {'中文特征':<40} {'英文翻译':<40} {'SHAP值':<10}\n")
            f.write('-'*100 + '\n')
            for i, (cn, en, idx) in enumerate(zip(top_cn, top_en, top_idx), 1):
                shap_val = shap_values[idx] if idx < len(shap_values) else 0
                f.write(f"{i:<5} {cn:<40} {en:<40} {shap_val:<10.6f}\n")
        print(f"\n✓ 对照表已保存到: {save_path}")

# ========== 主要处理流程 ==========
def main():
    print("="*60)
    print("SMOTE + RandomForest 文本二分类分析")
    print("="*60)
    
    # 1. 加载屏蔽词配置
    exact_blacklist, pattern_blacklist = load_blacklist_config()
    
    # 2. 读数据+分词+停词
    def read_excel(path):
        df = pd.read_excel(path, engine='openpyxl').iloc[:, 1:3]
        df.columns = ['text', 'label']
        return df.dropna(subset=['text', 'label'])
    
    with open('cn_stop.txt', encoding='utf-8') as f:
        stop = set(f.read().split())
    
    def cut(txt):
        return ' '.join([w for w in jieba.lcut(txt) if w not in stop])
    
    df = read_excel(EXCEL_FILE)
    df['text_cut'] = df['text'].astype(str).apply(cut)
    
    # 3. 向量化
    tfidf = TfidfVectorizer(lowercase=False, stop_words=None,
                            ngram_range=(1, 2), max_df=0.9, min_df=2)
    X = tfidf.fit_transform(df['text_cut'])
    y = df['label'].astype(int)
    feature_names = tfidf.get_feature_names_out()
    print(f"\n特征维度: {len(feature_names)}")
    
    # 4. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. SMOTE过采样
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # 6. 随机森林训练
    rf = RandomForestClassifier(n_estimators=600, max_depth=None,
                                min_samples_leaf=1, n_jobs=-1, random_state=42)
    rf.fit(X_res, y_res)
    
    # 7. 评估模型
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "="*60)
    print("模型性能评估")
    print('='*60)
    print(classification_report(y_test, y_pred, digits=4))
    
    # 8. 画图 - 类别分布
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pd.Series(y_train).value_counts().plot.bar(title="Before SMOTE", ax=ax[0], color='coral')
    pd.Series(y_res).value_counts().plot.bar(title="After SMOTE", ax=ax[1], color='skyblue')
    for a in ax: a.set_xticklabels(['0', '1'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '01_class_dist.png'), dpi=300)
    plt.close()
    print("✓ 类别分布图保存成功")
    
    # 9. SHAP分析（带屏蔽词过滤）
    print(f"\n{'='*60}")
    print("开始SHAP分析（带屏蔽词过滤）")
    print('='*60)
    
    # 9.1 采样
    test_idx = np.random.choice(X_test.shape[0], min(200, X_test.shape[0]), replace=False)
    X_test_sample_dense = X_test[test_idx].toarray()
    
    # 9.2 计算SHAP值
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_sample_dense, check_additivity=False)
    
    # 9.3 提取正类SHAP值
    if len(np.shape(shap_values)) == 3:
        shap_matrix = shap_values[:, :, 1]
    elif isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values
    
    print(f"SHAP矩阵形状: {shap_matrix.shape}")
    
    # 9.4 计算平均SHAP值
    feature_names_list = feature_names.tolist()
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    
    # 9.5 使用屏蔽词过滤特征
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
            print(f"{i+1:2d}. {feat_display:<60} SHAP: {item['shap']:.6f} ({item['reason']})")
    
    # 9.6 从过滤后的特征中选择Top-10（原来是30）
    TOP_N = 10  # 修改为10
    
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
    
    # 9.7 准备Top-10特征
    top_cn = [feature_names_list[i] for i in top_idx]
    top_shap = shap_matrix[:, top_idx]
    top_features = X_test_sample_dense[:, top_idx]
    
    # 9.8 翻译特征
    top_en = [translate_feature(feat, cn2en) for feat in top_cn]
    
    # 9.9 输出中英文对照表
    print_translation_table(
        top_cn, top_en, mean_abs_shap, top_idx,
        os.path.join(FIG_DIR, 'shap_translation_table.txt'),
        top_n=TOP_N
    )
    
    # 10. 绘制SHAP图
    # 10.1 Beeswarm图（英文）
    try:
        plt.figure(figsize=(8, 6))  # 调整图形大小以适应10个特征
        shap.summary_plot(top_shap, top_features,
                          feature_names=top_en,
                          plot_type="dot", show=False,
                          max_display=TOP_N)
        plt.title(f"Top-{TOP_N} SHAP Feature Importance (Beeswarm)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"02_feat_imp_shap_beeswarm_top{TOP_N}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Beeswarm图(英文)保存成功 (Top-{TOP_N})")
    except Exception as e:
        print(f"绘制beeswarm图时出错: {e}")
    
    # 10.2 Bar图（英文）
    try:
        plt.figure(figsize=(8, 6))  # 调整图形大小以适应10个特征
        shap.summary_plot(top_shap, top_features,
                          feature_names=top_en,
                          plot_type="bar", show=False,
                          max_display=TOP_N)
        plt.title(f"Top-{TOP_N} SHAP Feature Importance (Bar Chart)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"02_feat_imp_shap_bar_EN_top{TOP_N}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Bar图(英文)保存成功 (Top-{TOP_N})")
    except Exception as e:
        print(f"绘制bar图时出错: {e}")
    
    # 10.3 可选：绘制中文标签的图
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(top_shap, top_features,
                          feature_names=top_cn,
                          plot_type="dot", show=False,
                          max_display=TOP_N)
        plt.title(f"Top-{TOP_N} SHAP 特征重要性 (Beeswarm - 中文)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"02_feat_imp_shap_beeswarm_CN_top{TOP_N}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Beeswarm图(中文)保存成功 (Top-{TOP_N})")
    except Exception as e:
        print(f"绘制中文beeswarm图时出错: {e}")
    
    # 11. 其他评估图表
    # 11.1 ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC – SMOTE+RF')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, '03_roc.png'), dpi=300)
    plt.close()
    print("✓ ROC曲线保存成功")
    
    # 11.2 PR曲线
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1,
                label=f'Baseline = {y_test.mean():.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR – SMOTE+RF')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, '04_pr.png'), dpi=300)
    plt.close()
    print("✓ PR曲线保存成功")
    
    # 11.3 学习曲线
    train_sizes, train_scr, val_scr = learning_curve(
        rf, X_res, y_res, cv=3, scoring='f1', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1, 5), random_state=42)
    plt.figure(figsize=(5, 4))
    plt.plot(train_sizes, train_scr.mean(1), marker='o', label='train F1')
    plt.plot(train_sizes, val_scr.mean(1), marker='o', label='val F1')
    plt.xlabel('Training examples'); plt.ylabel('F1')
    plt.title('Learning Curve – SMOTE+RF')
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(FIG_DIR, '05_learning_curve.png'), dpi=300)
    plt.close()
    print("✓ 学习曲线保存成功")
    
    # 11.4 混淆矩阵
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix (%) – SMOTE+RF')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(FIG_DIR, '06_cm_norm.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix (Counts) – SMOTE+RF')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '06_cm_raw.png'), dpi=300)
    plt.close()
    print("✓ 混淆矩阵保存成功")
    
    # 12. 保存模型
    joblib.dump(rf,   os.path.join(MODEL_DIR, 'rf_smote.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    print(f"\n✓ 模型已保存到 {MODEL_DIR}/ 目录")
    
    # 13. 交叉验证和决策路径分析
    cv_imp = []
    for seed in range(3):
        rf_cv = RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1)
        rf_cv.fit(X_res, y_res)
        cv_imp.append(rf_cv.feature_importances_)
    
    imp_df   = pd.DataFrame(cv_imp, columns=feature_names)
    imp_mean = imp_df.mean().sort_values(ascending=False)
    imp_std  = imp_df.std()
    top_word = imp_mean.index[0]
    mean_val = imp_mean.iloc[0]
    std_val  = imp_std[top_word]
    
    print(f"\n交叉验证结果:")
    print(f"  '{top_word}' 平均重要性: {mean_val:.3f} ± {std_val:.3f}，位列第一")
    
    # 显示Top-10重要性特征
    print(f"\nTop-{TOP_N} 特征重要性:")
    print("-" * 80)
    for i, (feature, importance) in enumerate(imp_mean.head(TOP_N).items(), 1):
        print(f"{i:2d}. {feature:<40} 重要性: {importance:.6f}")
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"所有图表已保存到: {FIG_DIR}/")
    print(f"SHAP分析使用Top-{TOP_N}特征")
    print('='*60)

if __name__ == "__main__":
    main()