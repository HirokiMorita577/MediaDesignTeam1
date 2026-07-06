"""
パラメータ比較グラフ生成スクリプト
================================================================
参考スライドの形式に合わせ、1つのパラメータを変化させた複数の
学習曲線を同一グラフに重ねて比較します。

生成グラフ:
  compare_epsilon_decay.png   : ε decay を変えた比較
  compare_gamma.png           : γ（割引率）を変えた比較
  compare_lr.png              : α（学習率）を変えた比較
  compare_dqn_vs_sarsa.png    : DQN vs SARSA（同一パラメータ）

使い方:
  python compare_params.py                  # 全グラフ生成
  python compare_params.py --target epsilon_decay
  python compare_params.py --target gamma
  python compare_params.py --target lr
  python compare_params.py --target algorithm
  python compare_params.py --episodes 1000  # エピソード数を変更
================================================================
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from pacman_env import PacManEnv
from dqn_agent  import DQNAgent
from sarsa_agent import SARSAAgent

SAVE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(SAVE_DIR, ".cache_params")
os.makedirs(CACHE_DIR, exist_ok=True)

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


# ════════════════════════════════════════════════════════════
# 訓練ループ（共通）
# ════════════════════════════════════════════════════════════

def train_one(label, algorithm, episodes, lr, gamma, epsilon_decay,
              epsilon_end=0.025, epsilon_start=1.0, batch_size=128,
              ghost_mode="classic"):
    """1条件を訓練してスコア・ε履歴を返す（キャッシュ付き）"""

    import re
    safe_label = re.sub(r"[^\w]", "_", label, flags=re.ASCII)
    key = (f"{safe_label}_{algorithm}_{ghost_mode}_ep{episodes}_lr{lr}_g{gamma}"
           f"_ed{epsilon_decay}_ee{epsilon_end}").replace(".", "p")
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            d = json.load(f)
        print(f"  キャッシュ読込: {label}")
        return d["scores"], d["epsilons"]

    env = PacManEnv(render_mode=None, ghost_mode=ghost_mode)
    obs_size = env.observation_space.shape[0]

    if algorithm == "sarsa":
        agent = SARSAAgent(
            obs_size=obs_size, n_actions=4,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay, batch_size=batch_size,
        )
    else:  # dqn
        agent = DQNAgent(
            obs_size=obs_size, n_actions=4,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay, batch_size=batch_size,
            target_update_freq=300,
        )

    scores, epsilons = [], []
    t0 = time.time()
    print(f"  訓練中: {label}  ({algorithm.upper()}, ghost={ghost_mode}, {episodes}ep)")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        if algorithm == "sarsa":
            action = agent.select_action(obs, training=True)

        while True:
            if algorithm == "dqn":
                action = agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.push(obs, action, reward, next_obs, float(terminated or truncated))
                agent.train_step()
            else:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_obs, training=True)
                agent.push(obs, action, reward, next_obs, next_action, float(done))
                agent.train_step()
                action = next_action
            obs = next_obs
            if terminated or truncated:
                break

        scores.append(info["score"])
        epsilons.append(agent.epsilon)

        if ep % 200 == 0:
            avg = np.mean(scores[-100:])
            print(f"    Ep {ep:4d}/{episodes}  Avg100={avg:5.0f}  e={agent.epsilon:.4f}")

    env.close()
    print(f"    完了 ({time.time()-t0:.0f}s)  Best={max(scores)}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "epsilons": epsilons}, f)

    return scores, epsilons


# ════════════════════════════════════════════════════════════
# グラフ描画（共通コア）
# ════════════════════════════════════════════════════════════

COLORS = ["steelblue", "darkorange", "green", "crimson",
          "mediumpurple", "saddlebrown", "deeppink"]

def plot_comparison(runs, title, xlabel, ylabel_score, param_col_name,
                    window, save_path, show_epsilon=True):
    """
    runs: list of dict
      {
        "label"  : 凡例テキスト,
        "scores" : list[float],
        "epsilons": list[float],
        "param_val": パラメータ値（表示用）,
      }
    """
    n_eps = len(runs[0]["scores"])
    x = np.arange(1, n_eps + 1)

    if show_epsilon:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.subplots_adjust(hspace=0.08)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))

    # ── スコア軸 ──
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        scores = np.array(run["scores"])

        # 生データ（薄く）
        ax1.plot(x, scores, color=color, alpha=0.15, linewidth=0.5)

        # 移動平均
        w = min(window, len(scores))
        ma = np.convolve(scores, np.ones(w)/w, mode="valid")
        x_ma = np.arange(w, len(scores) + 1)
        ax1.plot(x_ma, ma, color=color, linewidth=2.2, label=run["label"])

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax1.set_ylabel(ylabel_score, fontsize=13)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=12, loc="upper left", framealpha=0.92)
    ax1.set_xlim(1, n_eps)
    if not show_epsilon:
        ax1.set_xlabel(xlabel, fontsize=13)

    # ── ε（探索率）軸 ──
    if show_epsilon:
        ax1.set_xticklabels([])
        for i, run in enumerate(runs):
            color = COLORS[i % len(COLORS)]
            ax2.plot(x, run["epsilons"], color=color, linewidth=1.5, alpha=0.85)
        ax2.set_xlabel(xlabel, fontsize=13)
        ax2.set_ylabel("ε（探索率）", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.25)
        ax2.set_xlim(1, n_eps)

    # ── 右上にパラメータ表 ──
    _add_param_table(ax1, runs, param_col_name)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {os.path.basename(save_path)}")


def _add_param_table(ax, runs, param_col_name):
    """グラフ右上にパラメータ一覧テキストを描画"""
    lines = [f"{'条件':<6}  {param_col_name}"]
    lines.append("─" * 28)
    for run in runs:
        lines.append(f"{run['label']:<14}  {run.get('param_val', '')}")
    text = "\n".join(lines)
    ax.text(0.98, 0.97, text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      alpha=0.85, edgecolor="gray"))


# ════════════════════════════════════════════════════════════
# 比較パターン定義
# ════════════════════════════════════════════════════════════

def compare_epsilon_decay(episodes, base_lr, base_gamma, window):
    """ε decay（探索率減衰）を変えた比較"""
    configs = [
        ("ε_decay = 0.990", 0.990),
        ("ε_decay = 0.995", 0.995),
        ("ε_decay = 0.999", 0.999),
    ]
    print("\n── ε decay 比較 ──")
    runs = []
    for label, ed in configs:
        s, e = train_one(label, "dqn", episodes, base_lr, base_gamma, ed)
        runs.append({"label": label, "scores": s, "epsilons": e,
                     "param_val": str(ed)})

    plot_comparison(
        runs,
        title=f"ε decay 比較  (α={base_lr}, γ={base_gamma}, {episodes}ep)",
        xlabel="エピソード数",
        ylabel_score="スコア（移動平均）",
        param_col_name="ε_decay",
        window=window,
        save_path=os.path.join(SAVE_DIR, "compare_epsilon_decay.png"),
    )


def compare_gamma(episodes, base_lr, base_eps_decay, window):
    """γ（割引率）を変えた比較"""
    configs = [
        ("γ = 0.80", 0.80),
        ("γ = 0.90", 0.90),
        ("γ = 0.95", 0.95),
        ("γ = 0.99", 0.99),
    ]
    print("\n── γ（割引率）比較 ──")
    runs = []
    for label, g in configs:
        s, e = train_one(label, "dqn", episodes, base_lr, g, base_eps_decay)
        runs.append({"label": label, "scores": s, "epsilons": e,
                     "param_val": str(g)})

    plot_comparison(
        runs,
        title=f"γ（割引率）比較  (α={base_lr}, ε_decay={base_eps_decay}, {episodes}ep)",
        xlabel="エピソード数",
        ylabel_score="スコア（移動平均）",
        param_col_name="γ",
        window=window,
        save_path=os.path.join(SAVE_DIR, "compare_gamma.png"),
    )


def compare_lr(episodes, base_gamma, base_eps_decay, window):
    """α（学習率）を変えた比較"""
    configs = [
        ("α = 0.0001", 0.0001),
        ("α = 0.001",  0.001),
        ("α = 0.002",  0.002),
        ("α = 0.01",   0.01),
    ]
    print("\n── α（学習率）比較 ──")
    runs = []
    for label, lr in configs:
        s, e = train_one(label, "dqn", episodes, lr, base_gamma, base_eps_decay)
        runs.append({"label": label, "scores": s, "epsilons": e,
                     "param_val": str(lr)})

    plot_comparison(
        runs,
        title=f"α（学習率）比較  (γ={base_gamma}, ε_decay={base_eps_decay}, {episodes}ep)",
        xlabel="エピソード数",
        ylabel_score="スコア（移動平均）",
        param_col_name="α（学習率）",
        window=window,
        save_path=os.path.join(SAVE_DIR, "compare_lr.png"),
    )


def compare_ghost_mode(episodes, base_lr, base_gamma, base_eps_decay, window):
    """ゴーストAI: classic（本家4体行動）vs bfs（全員直接追跡）"""
    print("\n── ゴーストAI 比較 ──")
    runs = []
    configs = [
        ("Classic (本家4体)", "classic"),
        ("BFS (全員直接追跡)", "bfs"),
    ]
    for label, gm in configs:
        s, e = train_one(label, "dqn", episodes,
                         base_lr, base_gamma, base_eps_decay,
                         ghost_mode=gm)
        runs.append({"label": label, "scores": s, "epsilons": e,
                     "param_val": gm})

    plot_comparison(
        runs,
        title=f"ゴーストAI比較: Classic vs BFS  "
              f"(α={base_lr}, γ={base_gamma}, ε_decay={base_eps_decay}, {episodes}ep)",
        xlabel="エピソード数",
        ylabel_score="スコア（移動平均）",
        param_col_name="ゴーストAI",
        window=window,
        save_path=os.path.join(SAVE_DIR, "compare_ghost_mode.png"),
    )


def compare_algorithm(episodes, base_lr, base_gamma, base_eps_decay, window):
    """DQN vs SARSA（同一パラメータで比較）"""
    print("\n── DQN vs SARSA 比較 ──")
    runs = []
    for alg in ["dqn", "sarsa"]:
        label = "Q学習 / DQN" if alg == "dqn" else "SARSA"
        s, e = train_one(label, alg, episodes,
                         base_lr, base_gamma, base_eps_decay)
        runs.append({"label": label, "scores": s, "epsilons": e,
                     "param_val": alg.upper()})

    plot_comparison(
        runs,
        title=f"Q学習(DQN) vs SARSA  "
              f"(α={base_lr}, γ={base_gamma}, ε_decay={base_eps_decay}, {episodes}ep)",
        xlabel="エピソード数",
        ylabel_score="スコア（移動平均）",
        param_col_name="アルゴリズム",
        window=window,
        save_path=os.path.join(SAVE_DIR, "compare_dqn_vs_sarsa.png"),
    )


# ════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════

def main(args):
    os.chdir(SAVE_DIR)
    window = min(100, args.episodes // 5)

    # 基本パラメータ（PSO最適値を採用）
    base_lr        = args.lr
    base_gamma     = args.gamma
    base_eps_decay = args.epsilon_decay

    print("=" * 60)
    print("  パラメータ比較グラフ生成")
    print(f"  エピソード数: {args.episodes}")
    print(f"  基本設定: α={base_lr}, γ={base_gamma}, ε_decay={base_eps_decay}")
    print("=" * 60)

    targets = args.target if args.target else \
              ["epsilon_decay", "gamma", "lr", "algorithm", "ghost_mode"]

    if "epsilon_decay" in targets:
        compare_epsilon_decay(args.episodes, base_lr, base_gamma, window)

    if "gamma" in targets:
        compare_gamma(args.episodes, base_lr, base_eps_decay, window)

    if "lr" in targets:
        compare_lr(args.episodes, base_gamma, base_eps_decay, window)

    if "algorithm" in targets:
        compare_algorithm(args.episodes, base_lr, base_gamma, base_eps_decay, window)

    if "ghost_mode" in targets:
        compare_ghost_mode(args.episodes, base_lr, base_gamma, base_eps_decay, window)

    print("\n完了。生成ファイル:")
    for fname in ["compare_epsilon_decay.png", "compare_gamma.png",
                  "compare_lr.png", "compare_dqn_vs_sarsa.png",
                  "compare_ghost_mode.png"]:
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):
            print(f"  {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="パラメータ比較グラフ生成")
    parser.add_argument("--episodes",      type=int,   default=500,
                        help="訓練エピソード数（デフォルト500）")
    parser.add_argument("--lr",            type=float, default=0.002,
                        help="基本学習率")
    parser.add_argument("--gamma",         type=float, default=0.93,
                        help="基本割引率")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="基本ε減衰率")
    parser.add_argument("--target",        type=str,   nargs="+",
                        choices=["epsilon_decay", "gamma", "lr", "algorithm", "ghost_mode"],
                        help="比較対象（省略時は全て生成）")
    args = parser.parse_args()
    main(args)
