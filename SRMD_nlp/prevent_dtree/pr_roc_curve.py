# pr_roc_curve.py
import joblib, os
from prevent_dtree import PreventRuleFeaturizer   # 类定义
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# ---- 1. 加载模型 ----
bundle = joblib.load('prevent_dtree.joblib')
clf   = bundle['clf']
featurizer = bundle['featurizer']
tfidf      = bundle['tfidf']

# ---- 2. 加载验证集（约定：dev.txt 一行一条，dev_label.txt 一行一个 0/1）----
dev_text  = open('demo/dev.txt', encoding='utf-8').read().splitlines()
dev_label = list(map(int, open('demo/dev_label.txt').read().split()))

# ---- 3. 复刻 build_features ----
def build_features(texts, tfidf, featurizer, fit=False):
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)
    X_rule = featurizer.transform(texts).values
    return hstack([X_tfidf, X_rule], format='csr')

X_dev = build_features(dev_text, tfidf, featurizer, fit=False)
y_score = clf.predict_proba(X_dev)[:, 1]

# ---- 4. PR 曲线 ----
precision, recall, _ = precision_recall_curve(dev_label, y_score)
pr_auc = auc(recall, precision)
# 计算正样本比例（随机水平）
pos_ratio = sum(dev_label) / len(dev_label)
plt.figure()
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.2f}', lw=2)
plt.axhline(y=pos_ratio, color='grey', linestyle='--',
            label=f'Random (p={pos_ratio:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('pr_curve.png', dpi=300)
plt.close()

# ---- 5. ROC 曲线 ----
fpr, tpr, _ = roc_curve(dev_label, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.2f}', lw=2)
plt.plot([0, 1], [0, 1], '--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

print('→ pr_curve.png & roc_curve.png 已保存')