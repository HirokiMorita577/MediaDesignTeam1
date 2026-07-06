"""
訓練ログCSVを重ねて比較グラフを生成するスクリプト
================================================================
train.py が出力する *_log.csv を複数指定して、学習曲線を1枚の
グラフに重ねて比較します。

使い方:
  # 2つのCSVを比較
  python plot_logs.py pacman_model_log.csv pacman_model_bfs_log.csv

  # ラベルを付けて比較
  python plot_logs.py pacman_model_log.csv pacman_model_bfs_log.csv \
      --labels "Classic" "BFS"

  # 出力ファイル名を指定
  python plot_logs.py *.csv --out compare_result.png

  # 移動平均の窓サイズを変更（デフォルト100）
  python plot_logs.py *.csv --window 200
================================================================
"""

import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob


# ── 日本語フォント ──
def _jp():
    cands = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAGothic", "Noto Sans CJK JP"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in avail:
            return c
    return None

JP = _jp()
if JP:
    plt.rcParams["font.family"] = JP

COLORS = ["steelblue", "darkorange", "green", "crimson",
          "mediumpurple", "saddlebrown", "deeppink"]


def load_csv(path):
    """CSVを読み込んでエピソード・スコア・εのリストを返す"""
    episodes, scores, epsilons = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            epsilons.append(float(row["epsilon"]))
    return episodes, scores, epsilons


def get_meta(path):
    """CSVの1行目からメタ情報を取得してラベル用文字列を返す"""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ghost = row.get("ghost_mode", "?")
            lr    = row.get("lr", "?")
            gamma = row.get("gamma", "?")
            ed    = row.get("epsilon_decay", "?")
            return f"ghost={ghost} α={lr} γ={gamma} ε_decay={ed}"
    return os.path.basename(path)


def plot_logs(csv_paths, labels, window, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    max_ep = 0
    for i, (path, label) in enumerate(zip(csv_paths, labels)):
        color = COLORS[i % len(COLORS)]
        eps, scores, epsilons = load_csv(path)
        x = np.array(eps)
        s = np.array(scores)
        e = np.array(epsilons)
        max_ep = max(max_ep, x[-1])

        # 生スコア（薄く）
        ax1.plot(x, s, color=color, alpha=0.15, linewidth=0.5)

        # 移動平均
        w = min(window, len(s))
        ma = np.convolve(s, np.ones(w)/w, mode="valid")
        x_ma = x[w-1:]
        ax1.plot(x_ma, ma, color=color, linewidth=2.2, label=label)

        # ε曲線
        ax2.plot(x, e, color=color, linewidth=1.5, alpha=0.85, label=label)

    ax1.set_title(f"学習曲線比較（移動平均 window={window}）",
                  fontsize=15, fontweight="bold", pad=12)
    ax1.set_ylabel("スコア", fontsize=13)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=11, loc="upper left", framealpha=0.92)
    ax1.set_xlim(1, max_ep)
    ax1.set_xticklabels([])

    ax2.set_xlabel("エピソード数", fontsize=13)
    ax2.set_ylabel("ε（探索率）", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(1, max_ep)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {out_path}")


def main(args):
    # glob展開（ワイルドカード対応）
    csv_paths = []
    for pattern in args.csv_files:
        matched = sorted(glob.glob(pattern))
        if matched:
            csv_paths.extend(matched)
        else:
            csv_paths.append(pattern)

    if not csv_paths:
        print("エラー: CSVファイルが見つかりません")
        return

    # ラベル: 指定があれば使用、なければファイル名＋メタ情報
    if args.labels:
        labels = args.labels
        # 足りない分はファイル名で補完
        while len(labels) < len(csv_paths):
            labels.append(os.path.basename(csv_paths[len(labels)]))
    else:
        labels = []
        for p in csv_paths:
            name = os.path.splitext(os.path.basename(p))[0].replace("_log", "")
            meta = get_meta(p)
            labels.append(f"{name}\n({meta})")

    print(f"比較対象 ({len(csv_paths)}件):")
    for p, l in zip(csv_paths, labels):
        first_line = l.split('\n')[0]
        print(f"  {first_line}  ← {p}")

    plot_logs(csv_paths, labels, args.window, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練ログCSV比較グラフ生成")
    parser.add_argument("csv_files", nargs="+",
                        help="比較するCSVファイル（ワイルドカード可: *_log.csv）")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="各CSVのラベル名（省略時はファイル名＋設定値）")
    parser.add_argument("--window", type=int, default=100,
                        help="移動平均の窓サイズ（デフォルト100）")
    parser.add_argument("--out", type=str, default="compare_logs.png",
                        help="出力ファイル名（デフォルト: compare_logs.png）")
    args = parser.parse_args()
    main(args)
