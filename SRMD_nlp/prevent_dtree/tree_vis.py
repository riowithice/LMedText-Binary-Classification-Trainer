# tree_vis.py   ← 直接覆盖旧文件
import os, sys, platform, subprocess
from pathlib import Path
import joblib
from sklearn.tree import export_graphviz
import pydotplus

# ---------- 1. 把训练脚本所在目录加入 PYTHONPATH ----------
train_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(train_dir))
from prevent_dtree import PreventRuleFeaturizer   # noqa: E402

# ---------- 2. 载入模型 ----------
bundle = joblib.load(train_dir / 'prevent_dtree.joblib')
clf = bundle['clf']
raw_names = list(bundle['tfidf'].get_feature_names_out()) + \
            list(bundle['featurizer'].transform(['']).columns)

# ---------- 3. 中英对照 ----------
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
    "颅骨未见明显骨折": "No_obvious_skull_fracture",
    "头皮血肿": "Scalp_haematoma",
    "神清": "Alert",
    "无恶心呕吐": "No_nausea_vomiting",
    "高血压": "hypertension",
}
feat_names_en = [feat_map.get(f, f) for f in raw_names]

# ---------- 4. 通用画图函数 ----------
def _make_png(dot_data, png_path, font_name=None):
    """
    把 dot 数据渲染成 PNG，并强制嵌入字体，防止方框。
    font_name=None  表示用默认字体（英文）
    """
    graph = pydotplus.graph_from_dot_data(dot_data)
    if font_name:
        # 对所有文本对象强制字体
        for node in graph.get_node_list():
            if node.get_attributes().get('label'):
                node.set('fontname', font_name)
        for edge in graph.get_edge_list():
            for k in ('label', 'fontname'):
                if edge.get_attributes().get(k):
                    edge.set(k, font_name)
    # 输出 PNG
    graph.write_png(png_path)
    print(f'✅  {png_path}  已生成')

# ---------- 5. 生成英文图 ----------
dot_en = export_graphviz(
    clf, max_depth=3, feature_names=feat_names_en,
    class_names=['NonPrev', 'Prev'],
    proportion=True, rounded=True, filled=True
)
_make_png(dot_en, 'tree_english.png')

# ---------- 6. 生成中文图 ----------
dot_cn = export_graphviz(
    clf, max_depth=3, feature_names=raw_names,
    class_names=['非预防', '预防'],
    proportion=True, rounded=True, filled=True
)
# Windows 自带 SimHei，Linux/Mac 可用系统黑体或思源黑体
_make_png(dot_cn, 'tree_chinese.png',
          font_name='SimHei' if platform.system() == 'Windows' else 'WenQuanYi Zen Hei')