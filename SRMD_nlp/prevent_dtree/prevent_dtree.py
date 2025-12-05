#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预防用药决策树文本分类  单文件可运行
python prevent_dtree.py  --data_dir ./demo  --yml prevent_rules.yml
"""
import yaml, re, joblib, argparse, os, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.sparse import hstack
from sklearn.metrics import classification_report
import datetime as dt

from sklearn.metrics import precision_recall_curve


# ---------- 1. YAML 热加载 ----------
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

    # ===== 血压正常 =====
    def _bp_normal_flag(self, text: str) -> int:
        cfg = self.cfg.get('bp_normal')
        if not cfg:
            return 0
        m = re.search(cfg['pattern'], text, re.I)
        if not m:
            return 0
        try:
            sbp = int(m.group(1))   # 收缩压
            dbp = int(m.group(2))   # 舒张压
        except ValueError:
            return 0
        return int((sbp < cfg['threshold_systolic']) and (dbp < cfg['threshold_diastolic']))

    # ===== 1. 手术时长 >3h =====
    def _op_duration_flag(self, text: str) -> int:
        cfg = self.cfg.get('op_duration')
        if not cfg:
            return 0
        # 统一抓 2024-01-21 00:14 或 2024年01月21日 00时14分
        pattern = r'手术开始时间[:：]\s*(\d{4}[年-]\d{1,2}[月-]\d{1,2}[日\s]+\d{1,2}[：:]\d{1,2})\s*手术结束时间[:：]\s*(\d{4}[年-]\d{1,2}[月-]\d{1,2}[日\s]+\d{1,2}[：:]\d{1,2})'
        m = re.search(pattern, text)
        if not m:
            return 0
        try:
            # 统一转成 2024-01-21 00:14 格式再解析
            start_str = m.group(1).replace('年','-').replace('月','-').replace('日','').replace('时',':').replace('分','')
            end_str   = m.group(2).replace('年','-').replace('月','-').replace('日','').replace('时',':').replace('分','')
            start = dt.datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M")
            end   = dt.datetime.strptime(end_str.strip(),   "%Y-%m-%d %H:%M")
        except ValueError:
            return 0
        delta_hours = (end - start).total_seconds() / 3600
        return int(delta_hours > cfg.get('threshold', 3))

    def transform(self, texts):
        rows = []
        for t in texts:              # ← 去掉 .lower()
            t = t.lower()            # ← 这里 lower
            # ===== 否定检测：把“无/未见/否认...骨折/伤/感染...”整段抹掉 =====
            neg_pattern = re.compile(
                r'(无|未见|否认|未触及|未及|未闻及|不考虑|排除)\s*[\u4e00-\u9fa5]{0,8}?(骨折|伤|感染|出血|撕裂|挤压|囊肿|脓毒)',
                re.I
            )
            t = neg_pattern.sub('', t)
            row = {}
            # 正常 keyword → 权重 = 1
            for w in self.cfg['keyword']:
                row[f'kw_{w}'] = int(w in t)

            # 低权重词 → 权重 = 0.2（你自己定）
            for w in self.cfg.get('low_weight', []):
                row[f'lw_{w}'] = int(str(w) in t) * 0.2   # ← 手动降权重

            # === 物理删除：删掉含年份/日期/时间的列 ===
            drop_cols = [c for c in row.keys() if re.search(r'\d{4}-\d{1,2}-\d{1,2}|\d{1,2}:\d{1,2}|\d{4}年|\d{1,2}时|\d{1,2}分|\d{1,2}日', c)]
            for dc in drop_cols:
                del row[dc]

            # === 物理删除：删掉 kw_ 列且包含 shield 词的列 ===
            shield_words = self.cfg.get('shield', [])
            drop_cols = [c for c in row.keys() if c.startswith('kw_') and any(sh in c for sh in shield_words)]
            for dc in drop_cols:
                del row[dc]

            row['negation'] = any(neg in t for neg in self.cfg['negation'])
            rows.append(row)
        return pd.DataFrame(rows).astype(float)   # ← 注意 float

# ---------- 2. 特征工厂 ----------
def _shield_text(texts, shield_list):
    """把 shield_list 里的词全部从原文里抹掉"""
    if not shield_list:
        return texts
    # 拼一个正则，\b 防止误杀
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, shield_list)) + r')\b', flags=re.I)
    return [pattern.sub('', t) for t in texts]
# ---------- 2. 特征工厂 ----------
def build_features(texts, tfidf=None, featurizer=None, fit=False):
    """返回稀疏矩阵：TF-IDF + 规则"""
    # 1. 先脱敏
    texts = _shield_text(texts, getattr(featurizer, 'shield_words', []))
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)
    X_rule = featurizer.transform(texts).values
    return hstack([X_tfidf, X_rule], format='csr')


# ---------- 3. 训练 ----------
def train_model(texts, labels, yml_rules, model_path):
    # 2. 中文停用词表（github 搜“中文停用词表”存成 cn_stop.txt）
    with open('cn_stop.txt', encoding='utf-8') as f:
        stop = list(set(f.read().split()))
    tfidf = TfidfVectorizer(binary=True, ngram_range=(1,3),
                            max_features=30000, min_df=5, max_df=1.0, stop_words=stop)  
    # max_df原始0.8把参数调宽松1,原始min_df=5,增加stop_words=None, 
    #     | 参数             | 建议值范围       | 备注                |
    # | -------------- | ----------- | ----------------- |
    # | `min_df`       | 1–5         | 数据量 <1000 行时先用 1  |
    # | `max_df`       | 0.9–1.0     | 中文病历常见高频词不多，可 1.0 |
    # | `max_features` | 10000–30000 | 根据内存和速度权衡         |

    featurizer = PreventRuleFeaturizer(yml_rules)

    X = build_features(texts, tfidf, featurizer, fit=True)
    y = np.array(labels)

    #  ============平衡 + 降采样双保险（模型层）===================
    # from sklearn.utils import resample
    # # 1. 先保持原样本
    # X_raw, y_raw = X, y

    # # 2. 1:2 降采样（负:正 = 2:1）
    # neg_idx = np.where(y_raw == 0)[0]
    # pos_idx = np.where(y_raw == 1)[0]
    # neg_down = resample(neg_idx, n_samples=len(pos_idx) * 2, random_state=42)
    # new_idx  = np.concatenate([pos_idx, neg_down])
    # X_bal, y_bal = X_raw[new_idx], y_raw[new_idx]

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,  #放松剪枝（让树敢用高危词当大分叉）原来 60→20
                                 class_weight='balanced', random_state=42)

    # clf.fit(X_bal, y_bal)
    clf.fit(X, y)

    joblib.dump({'tfidf': tfidf, 'featurizer': featurizer, 'clf': clf}, model_path)
    print(f'模型已保存 → {model_path}')
    return clf, tfidf, featurizer


# ---------- 4. 评估 ----------
def eval_model(clf, tfidf, featurizer, texts, labels):
    X = build_features(texts, tfidf, featurizer, fit=False)
    proba = clf.predict_proba(X)[:, 1]   # 正类概率

    precision, recall, thresholds = precision_recall_curve(labels, proba)
    # 找最接近 recall=0.8 的阈值
    idx = np.argmin(np.abs(recall - 0.8))
    print("建议阈值：", thresholds[idx])

    # === 灰区加宽到 0.2-0.8 ===
    auto   = proba >= 0.7                     # 自动通过
    review = (0.2 <= proba) & (proba < 0.7)   # 人工复核
    deny   = proba < 0.2                      # 直接否定

    # === 业务流打印 ===
    auto_cases   = [texts[i] for i in range(len(texts)) if auto[i]]
    review_cases = [texts[i] for i in range(len(texts)) if review[i]]
    deny_cases   = [texts[i] for i in range(len(texts)) if deny[i]]
    print('=== 业务动作 ===')
    print(f'自动通过 : {len(auto_cases)} 条')
    print(f'人工复核 : {len(review_cases)} 条')
    print(f'直接否定 : {len(deny_cases)} 条')

    # === 最终判决 = 0.2（统一阈值） ===
    y_pred = (proba >= thresholds[idx]).astype(int)   # 0.5→0.2
    print(classification_report(labels, y_pred, target_names=['非预防', '预防'], zero_division=0))

    proba = clf.predict_proba(X)[:, 1]   # 重新拉正类概率
    print('高置信正例 :', sum(proba >= 0.7))
    print('灰区复核   :', sum((proba >= 0.2) & (proba < 0.7)))
    print('低置信     :', sum(proba < 0.2))
    return y_pred


# ---------- 5. 打印规则 ----------
def show_rules(clf, tfidf, featurizer, top=2000):
    feat_names = list(tfidf.get_feature_names_out()) + list(featurizer.transform(['']).columns)
    print(export_text(clf, feature_names=feat_names, decimals=0)[:top])


# ---------- 6. 预测单份病历 ----------
def predict_one(text, model_path):
    bundle = joblib.load(model_path)
    X = build_features([text], bundle['tfidf'], bundle['featurizer'], fit=False)
    proba = bundle['clf'].predict_proba(X)[0, 1]
    pred = int(proba > 0.3) # 先降到 0.3，原本0.5，是判定阈值
    return pred, proba


# ---------- 7. CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='文件夹：train.txt  dev.txt  label.txt')
    ap.add_argument('--yml', default='prevent_rules.yml', help='YAML 规则文件')
    ap.add_argument('--model', default='prevent_dtree.joblib')
    args = ap.parse_args()

    # 7.1 读数据（约定：train.txt 一行一份病历；label.txt 对应 0/1）
    train_text = Path(args.data_dir, 'train.txt').read_text(encoding='utf-8').splitlines()
    train_label = list(map(int, Path(args.data_dir, 'label.txt').read_text().split()))
    dev_text = Path(args.data_dir, 'dev.txt').read_text(encoding='utf-8').splitlines()
    dev_label = list(map(int, Path(args.data_dir, 'dev_label.txt').read_text().split()))

    # 7.2 训练
    clf, tfidf, feat = train_model(train_text, train_label, args.yml, args.model)

    # 7.3 评估
    eval_model(clf, tfidf, feat, dev_text, dev_label)

    # 7.4 打印规则
    show_rules(clf, tfidf, feat)

    n_nodes = clf.tree_.node_count
    n_leaves = clf.tree_.n_leaves
    max_depth = clf.tree_.max_depth
    print(f"总节点数：{n_nodes}，叶节点数：{n_leaves}，最大深度：{max_depth}")


if __name__ == '__main__':
    main()