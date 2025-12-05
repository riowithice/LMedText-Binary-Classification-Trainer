#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键把原始文本+标签拆成 train/dev 两份（分层随机）
python split_train_dev.py --ratio 0.8
"""
import argparse
import random
from pathlib import Path

def split_file(src_txt: Path, src_lbl: Path, ratio: float, seed: int = 42):
    random.seed(seed)

    # 读原始数据
    texts = src_txt.read_text(encoding='utf-8').splitlines()
    labels = src_lbl.read_text(encoding='utf-8').splitlines()

    if len(texts) != len(labels):
        raise ValueError('文本与标签行数不一致！')

    # 打包 → 分层随机打乱
    data = list(zip(texts, labels))
    random.shuffle(data)

    # 切分
    n_train = int(len(data) * ratio)
    train_data, dev_data = data[:n_train], data[n_train:]

    # 写回 4 个文件
    Path('train.txt').write_text('\n'.join([d[0] for d in train_data]), encoding='utf-8')
    Path('train_label.txt').write_text('\n'.join([d[1] for d in train_data]), encoding='utf-8')
    Path('dev.txt').write_text('\n'.join([d[0] for d in dev_data]), encoding='utf-8')
    Path('dev_label.txt').write_text('\n'.join([d[1] for d in dev_data]), encoding='utf-8')

    print(f'已完成！train={len(train_data)}  dev={len(dev_data)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--txt', default='原始文本.txt', help='原始文本文件')
    ap.add_argument('--lbl', default='原始标签.txt', help='原始标签文件')
    ap.add_argument('--ratio', type=float, default=0.8, help='train 占比 (0-1)')
    args = ap.parse_args()

    split_file(Path(args.txt), Path(args.lbl), args.ratio)