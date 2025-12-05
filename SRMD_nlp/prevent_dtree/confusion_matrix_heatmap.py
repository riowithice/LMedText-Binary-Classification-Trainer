# confusion_matrix_both.py
import joblib
from prevent_dtree import PreventRuleFeaturizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
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
pred    = (y_score >= 0.76).astype(int)

# ---------- 2. 中英标签 ----------
cm = confusion_matrix(dev_label, pred)          # 2×2 矩阵
label_cn = ['非预防', '预防']
label_en = ['NonPrev', 'Prev']

# 3. 自动找中文字体
cn_fonts = ['Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
font_usable = next((f for f in cn_fonts if f in [x.name for x in fm.fontManager.ttflist]), 'SimSun')

# ---------- 4. 中文热力图 ----------
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_cn, yticklabels=label_cn)
plt.ylabel('Actual', fontproperties=fm.FontProperties(fname=fm.findfont(font_usable)))
plt.xlabel('Predicted', fontproperties=fm.FontProperties(fname=fm.findfont(font_usable)))
plt.title('混淆矩阵 (thr=0.76)', fontproperties=fm.FontProperties(fname=fm.findfont(font_usable)))
plt.tight_layout()
plt.savefig('confusion_matrix_cn.png', dpi=300)
plt.close()

# ---------- 5. 英文热力图 ----------
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_en, yticklabels=label_en)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.title('Confusion Matrix (thr=0.76)')
plt.tight_layout()
plt.savefig('confusion_matrix_en.png', dpi=300)
plt.close()

print('→ confusion_matrix_cn.png & confusion_matrix_en.png 已保存')