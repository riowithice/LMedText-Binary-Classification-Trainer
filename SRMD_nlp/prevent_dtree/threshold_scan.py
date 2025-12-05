# threshold_scan.py
import joblib, numpy as np, matplotlib.pyplot as plt
from prevent_dtree import PreventRuleFeaturizer   # 提供类定义
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import hstack

# ---------- 1. 加载模型 & 验证集 ----------
bundle = joblib.load('prevent_dtree.joblib')
clf   = bundle['clf']
featurizer = bundle['featurizer']
tfidf      = bundle['tfidf']

dev_text  = open('demo/dev.txt', encoding='utf-8').read().splitlines()
dev_label = list(map(int, open('demo/dev_label.txt').read().split()))

def build_features(texts, tfidf, featurizer, fit=False):
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)
    X_rule = featurizer.transform(texts).values
    return hstack([X_tfidf, X_rule], format='csr')

X_dev = build_features(dev_text, tfidf, featurizer, fit=False)
y_score = clf.predict_proba(X_dev)[:, 1]          # ← 生成预测概率

# ---------- 2. 阈值扫描 ----------
thr = np.linspace(0, 1, 101)
prec, rec, f1 = [], [], []
for t in thr:
    pred = (y_score >= t).astype(int)
    prec.append(precision_score(dev_label, pred, zero_division=0))
    rec.append(recall_score(dev_label, pred, zero_division=0))
    f1.append(f1_score(dev_label, pred, zero_division=0))

# ---------- 3. 画图 ----------
plt.figure(figsize=(7, 4))
plt.plot(thr, prec, label='Precision')
plt.plot(thr, rec,  label='Recall')
plt.plot(thr, f1,   label='F1', lw=2)
plt.axvline(0.76, color='red', ls='--', label='current thr=0.76')
plt.xlabel('Threshold'); plt.ylabel('Score')
plt.title('Threshold vs. Metrics'); plt.legend()
plt.grid(); plt.tight_layout()
plt.savefig('threshold_scan.png', dpi=300)
print('→ threshold_scan.png 已保存')