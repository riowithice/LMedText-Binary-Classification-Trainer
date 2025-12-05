# calibration_curve.py
import joblib
from prevent_dtree import PreventRuleFeaturizer
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
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

X_dev   = build_features(dev_text, tfidf, featurizer, fit=False)
y_score = clf.predict_proba(X_dev)[:, 1]

# ---------- 2. 校准曲线 ----------
# n_bins=10 把 0-1 分成 10 桶，看每桶里真实阳性比例 vs 平均预测概率
frac_pos, mean_pred = calibration_curve(dev_label, y_score, n_bins=10)

# ---------- 3. 画图 ----------
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o', label='Model')
plt.plot([0, 1], [0, 1], '--', color='grey', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('calibration.png', dpi=300)
print('→ calibration.png 已保存')