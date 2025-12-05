#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案3：SMOTE + 降采样 + 全套可视化
python prevent_dtree_smote.py --data_dir ./demo --yml prevent_rules.yml
"""
import yaml, re, joblib, argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from scipy.sparse import hstack
from sklearn.metrics import (classification_report, precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score,
                             accuracy_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
import datetime as dt
#import scikitplot as skplt   # 一键 ROC/PR/Calibration

# ---------- 1. 热加载规则（与原文件一致，略） ----------
class PreventRuleFeaturizer:
    def __init__(self, yml_path):
        self.cfg = yaml.safe_load(Path(yml_path).read_text(encoding='utf-8'))
        self._re = [re.compile(p, re.I) for p in self.cfg['regex']]
        self.shield_words = self.cfg.get('shield', [])

    def _platelet_low_flag(self, text: str) -> int:
        cfg = self.cfg.get('platelet_low')
        if not cfg:
            return 0
        m = re.search(cfg['pattern'], text, re.I)
        if not m:
            return 0
        try:
            val = int(m.group(1))
        except ValueError:
            return 0
        return int(val < cfg['threshold'])

    def _bp_normal_flag(self, text: str) -> int:
        cfg = self.cfg.get('bp_normal')
        if not cfg:
            return 0
        m = re.search(cfg['pattern'], text, re.I)
        if not m:
            return 0
        try:
            sbp = int(m.group(1))
            dbp = int(m.group(2))
        except ValueError:
            return 0
        return int((sbp < cfg['threshold_systolic']) and (dbp < cfg['threshold_diastolic']))

    def _op_duration_flag(self, text: str) -> int:
        cfg = self.cfg.get('op_duration')
        if not cfg:
            return 0
        pattern = r'手术开始时间[:：]\s*(\d{4}[年-]\d{1,2}[月-]\d{1,2}[日\s]+\d{1,2}[：:]\d{1,2})\s*手术结束时间[:：]\s*(\d{4}[年-]\d{1,2}[月-]\d{1,2}[日\s]+\d{1,2}[：:]\d{1,2})'
        m = re.search(pattern, text)
        if not m:
            return 0
        try:
            start_str = m.group(1).replace('年', '-').replace('月', '-').replace('日', '').replace('时', ':').replace('分', '')
            end_str = m.group(2).replace('年', '-').replace('月', '-').replace('日', '').replace('时', ':').replace('分', '')
            start = dt.datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M")
            end = dt.datetime.strptime(end_str.strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            return 0
        delta_hours = (end - start).total_seconds() / 3600
        return int(delta_hours > cfg.get('threshold', 3))

    def transform(self, texts):
        rows = []
        for t in texts:
            t = t.lower()
            neg_pattern = re.compile(
                r'(无|未见|否认|未触及|未及|未闻及|不考虑|排除)\s*[\u4e00-\u9fa5]{0,8}?(骨折|伤|感染|出血|撕裂|挤压|脓毒)', re.I)#|囊肿
            t = neg_pattern.sub('', t)
            row = {}
            for w in self.cfg['keyword']:
                row[f'kw_{w}'] = int(w in t)
            for w in self.cfg.get('low_weight', []):
                row[f'lw_{w}'] = int(str(w) in t) * 0.2
            drop_cols = [c for c in row.keys() if re.search(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}:\d{1,2}|\d{4}年|\d{1,2}时|\d{1,2}分|\d{1,2}日', c)]
            for dc in drop_cols:
                del row[dc]
            shield_words = self.cfg.get('shield', [])
            drop_cols = [c for c in row.keys() if c.startswith('kw_') and any(sh in c for sh in shield_words)]
            for dc in drop_cols:
                del row[dc]
            row['negation'] = any(neg in t for neg in self.cfg['negation'])
            rows.append(row)
        return pd.DataFrame(rows).astype(float)

def _shield_text(texts, shield_list):
    if not shield_list:
        return texts
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, shield_list)) + r')\b', flags=re.I)
    return [pattern.sub('', t) for t in texts]

def build_features(texts, tfidf=None, featurizer=None, fit=False):
    texts = _shield_text(texts, getattr(featurizer, 'shield_words', []))
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)
    X_rule = featurizer.transform(texts).values
    return hstack([X_tfidf, X_rule], format='csr')

# ---------- 2. SMOTE + 降采样 ----------
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def train_model(texts, labels, yml_rules, model_path):
    with open('cn_stop.txt', encoding='utf-8') as f:
        stop = list(set(f.read().split()))
    tfidf = TfidfVectorizer(binary=True, ngram_range=(1, 3), max_features=30000,
                            min_df=5, max_df=1.0, stop_words=stop)
    featurizer = PreventRuleFeaturizer(yml_rules)
    X = build_features(texts, tfidf, featurizer, fit=True)
    y = np.array(labels)

    # SMOTE + Under
    pipe = ImbPipeline(steps=[
        ('over', SMOTE(k_neighbors=2, random_state=42)),
        # ('over', SMOTE(k_neighbors=5, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy='auto', random_state=42))
    ])
    X_bal, y_bal = pipe.fit_resample(X, y)

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                 class_weight='balanced', random_state=42)
    clf.fit(X_bal, y_bal)
    joblib.dump({'tfidf': tfidf, 'featurizer': featurizer, 'clf': clf}, model_path)
    print(f'模型已保存 → {model_path}')
    return clf, tfidf, featurizer

# ---------- 3. 评估 + 6 张图 ----------
def eval_plots(clf, tfidf, featurizer, texts, labels, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    X = build_features(texts, tfidf, featurizer, fit=False)
    proba = clf.predict_proba(X)[:, 1]
    y = np.array(labels)

    # 1. PR 曲线
    precision, recall, thresh = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:0.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.savefig(f'{out_dir}/pr_curve.png', dpi=300)
    plt.close()

    # 2. ROC 曲线
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:0.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{out_dir}/roc_curve.png', dpi=300)
    plt.close()

    # 3. Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(y, proba, n_bins=10)
    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.savefig(f'{out_dir}/calibration.png', dpi=300)
    plt.close()

    # 4. 阈值扫描
    f1_vec = [f1_score(y, (proba >= t).astype(int)) for t in thresh]
    plt.figure(figsize=(6, 4))
    plt.plot(thresh, precision[:-1], label='Precision')
    plt.plot(thresh, recall[:-1], label='Recall')
    plt.plot(thresh, f1_vec, label='F1', lw=2, color='g')
    # 最佳阈值竖线
    best_thresh = thresh[np.argmax(f1_vec)]
    plt.axvline(best_thresh, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Scan')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/threshold_scan.png', dpi=300)
    plt.close()

    # 5. 决策树 pdf（需要系统装 graphviz）
    feat_names = list(tfidf.get_feature_names_out()) + list(featurizer.transform(['']).columns)
    export_graphviz(clf, out_file=f'{out_dir}/tree.dot',
                    feature_names=feat_names, max_depth=3, filled=True, rounded=True)
    os.system(f'dot -Tpng {out_dir}/tree.dot -o {out_dir}/decision_tree.png')

    # 6. 热力图：顶级特征重要性
    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[-30:]
    top_names = [feat_names[i] for i in top_idx]
    plt.figure(figsize=(6, 8))
    sns.heatmap(imp[top_idx].reshape(-1, 1), annot=True, fmt='.3f',
                yticklabels=top_names, xticklabels=['Importance'], cmap='viridis')
    plt.title('Top30 Feature Importance')
    plt.savefig(f'{out_dir}/feat_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'6 张图已保存至 ./{out_dir}/')

    # ===== 额外：输出 Accuracy、Recall、F1 =====
    # from sklearn.metrics import accuracy_score, f1_score, recall_score
    # 计算每个阈值的 F1
    f1_list = [f1_score(y, (proba >= t).astype(int)) for t in thresh]
    best_thresh = thresh[np.argmax(f1_list)]
    y_pred = (proba >= best_thresh).astype(int)
    print('\n=== 最佳阈值 = %.3f 时的指标 ===' % best_thresh)
    print('Accuracy : %.4f' % accuracy_score(y, y_pred))
    print('Recall   : %.4f' % recall_score(y, y_pred))
    print('F1-score : %.4f' % f1_score(y, y_pred))
    print(classification_report(y, y_pred, target_names=['非预防', '预防'], digits=4))

# ---------- 4. main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--yml', default='prevent_rules.yml')
    ap.add_argument('--model', default='prevent_dtree_smote.joblib')
    args = ap.parse_args()

    train_text = Path(args.data_dir, 'train.txt').read_text(encoding='utf-8').splitlines()
    train_label = list(map(int, Path(args.data_dir, 'label.txt').read_text().split()))
    dev_text = Path(args.data_dir, 'dev.txt').read_text(encoding='utf-8').splitlines()
    dev_label = list(map(int, Path(args.data_dir, 'dev_label.txt').read_text().split()))

    clf, tfidf, feat = train_model(train_text, train_label, args.yml, args.model)
    eval_plots(clf, tfidf, feat, dev_text, dev_label)

if __name__ == '__main__':
    main()