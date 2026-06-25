"""
学習曲線グラフの描画・保存
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUIなしで保存
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


def _get_jp_font():
    """日本語フォントを探す (なければデフォルト)"""
    candidates = [
        "MS Gothic", "Yu Gothic", "Meiryo", "IPAGothic",
        "Noto Sans CJK JP", "TakaoPGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


def plot_learning_curve(scores, title="AI Learning Curve (DQN)",
                        window=100, save_path="learning_curve.png",
                        color="steelblue"):
    """
    scores: エピソードごとのスコアリスト
    window: 移動平均ウィンドウ幅
    """
    jp_font = _get_jp_font()
    if jp_font:
        plt.rcParams["font.family"] = jp_font

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    episodes = np.arange(1, len(scores) + 1)

    # 各エピソードのスコア (薄い帯)
    ax.plot(episodes, scores,
            color=color, alpha=0.25, linewidth=0.6,
            label="各回のスコア")

    # 移動平均
    if len(scores) >= window:
        ma = np.convolve(scores, np.ones(window)/window, mode="valid")
        ax.plot(np.arange(window, len(scores)+1), ma,
                color=color, linewidth=2.2,
                label=f"{window}回移動平均")

    # ゼロライン
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_title("AIの学習過程", fontsize=14, pad=10)
    ax.set_xlabel("トレーニング回数", fontsize=11)
    ax.set_ylabel("スコア", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, len(scores))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curve saved: {save_path}")
    return save_path
