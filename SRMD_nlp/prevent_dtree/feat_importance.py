# feat_importance_both_fixcn.py
import joblib
from prevent_dtree import PreventRuleFeaturizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pydotplus
from sklearn.tree import export_graphviz
import os

# 1. 加载模型
bundle = joblib.load('prevent_dtree.joblib')
clf   = bundle['clf']
raw_names = list(bundle['tfidf'].get_feature_names_out()) + \
            list(bundle['featurizer'].transform(['']).columns)

# 2. 中英对照表
feat_map = {
    "kw_骨折": "kw_fracture",
    "kw_烧伤": "kw_burn",
    "kw_囊肿": "kw_cyst",
    "kw_伤": "kw_injury",
    "kw_抗凝": "kw_anticoag",
    "platelet_low": "platelet_low",
    "bp_normal": "bp_normal",
    "negation": "negation",
    "双下肢无水肿": "no_bilateral_edema",
}

# 3. 取重要性
topN = 30
imp = clf.feature_importances_
idx = imp.argsort()[::-1][:topN]

# 4. 构造 DataFrame
df_cn = pd.DataFrame({'feature': [raw_names[i] for i in idx], 'importance': imp[idx]})
df_en = pd.DataFrame({'feature': [feat_map.get(raw_names[i], raw_names[i]) for i in idx],
                      'importance': imp[idx]})

# 5. 自动找中文字体（防方框）
cn_fonts = ['Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
font_usable = next((f for f in cn_fonts if f in [x.name for x in fm.fontManager.ttflist]), 'SimSun')

# 6. 中文图
plt.figure(figsize=(6, 8))
sns.barplot(x='importance', y='feature', data=df_cn, color='steelblue')
plt.title('Top30 特征重要性（中文）', fontproperties=fm.FontProperties(fname=fm.findfont(font_usable)))
plt.tight_layout()
plt.savefig('feat_importance_cn.png', dpi=300)
plt.close()

# 7. 英文图
plt.figure(figsize=(6, 8))
sns.barplot(x='importance', y='feature', data=df_en, color='darkgreen')
plt.title('Top30 Feature Importance (English)')
plt.tight_layout()
plt.savefig('feat_importance_en.png', dpi=300)
plt.close()

print('✅ feat_importance_cn.png & feat_importance_en.png 已保存')