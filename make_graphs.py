"""
グラフ自動生成スクリプト
================================================================
参考資料のスライドに合わせた以下のグラフを自動生成・保存します:

  graph1_learning_curve_dqn_vs_sarsa.png
      Q学習 vs SARSA の学習曲線（スコア＋ε変化の二軸グラフ）

  graph2_pso_convergence.png
      PSO収束グラフ（各粒子のpbest + 群れのgbest を表示）

  graph3_pso_param_trajectory.png
      PSO最適化中のパラメータ軌跡（学習率・epsilon・gamma）

  graph4_final_comparison.png
      DQN vs SARSA 統計比較（平均・最大・後半平均）

使い方:
  python make_graphs.py               # 全グラフ生成（デフォルト300ep）
  python make_graphs.py --episodes 600
  python make_graphs.py --skip-train  # 既存の scores ファイルを使う
================================================================
"""

import argparse
import time
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

from pacman_env import PacManEnv
from dqn_agent  import DQNAgent
from sarsa_agent import SARSAAgent
from pso_optimizer import PSO
from pso_tuning import evaluate, decode_params, PARAM_NAMES, BOUNDS


# ── 日本語フォント ──
def _jp_font():
    cands = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAGothic", "Noto Sans CJK JP"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in avail:
            return c
    return None

JP = _jp_font()
if JP:
    plt.rcParams["font.family"] = JP

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════
# 1. 訓練ループ
# ════════════════════════════════════════════════════════════

def _train_agent(agent_type, episodes, lr, epsilon_decay, gamma, epsilon_end,
                 model_save_path):
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]

    if agent_type == "dqn":
        agent = DQNAgent(
            obs_size=obs_size, n_actions=4,
            lr=lr, gamma=gamma,
            epsilon_start=1.0, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=128, target_update_freq=300,
        )
    else:
        agent = SARSAAgent(
            obs_size=obs_size, n_actions=4,
            lr=lr, gamma=gamma,
            epsilon_start=1.0, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=128,
        )

    scores, epsilons, best = [], [], 0
    t0 = time.time()
    label = agent_type.upper()
    print(f"\n[{label}] 訓練開始  {episodes}ep")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()

        if agent_type == "sarsa":
            action = agent.select_action(obs, training=True)

        while True:
            if agent_type == "dqn":
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

        if info["score"] > best:
            best = info["score"]
            agent.save(model_save_path)

        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            bar = "#" * int(avg / 20)
            print(f"  [{label}] Ep {ep:4d}/{episodes}  Avg={avg:5.0f} {bar:<20}  "
                  f"Best={best}  e={agent.epsilon:.3f}")

    env.close()
    print(f"  [{label}] 完了 ({time.time()-t0:.1f}s)  Best={best}")
    return scores, epsilons


def run_training(episodes, lr, epsilon_decay, gamma, epsilon_end, cache_path):
    """DQN と SARSA を訓練してスコアを返す。キャッシュがあれば読み込む"""
    if os.path.exists(cache_path):
        print(f"キャッシュ読み込み: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            d = json.load(f)
        return d["dqn_scores"], d["sarsa_scores"], d["dqn_eps"], d["sarsa_eps"]

    dqn_scores,   dqn_eps   = _train_agent("dqn",   episodes, lr, epsilon_decay,
                                            gamma, epsilon_end, "model_dqn_graph.pt")
    sarsa_scores, sarsa_eps = _train_agent("sarsa", episodes, lr, epsilon_decay,
                                            gamma, epsilon_end, "model_sarsa_graph.pt")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"dqn_scores": dqn_scores, "sarsa_scores": sarsa_scores,
                   "dqn_eps": dqn_eps, "sarsa_eps": sarsa_eps}, f)
    return dqn_scores, sarsa_scores, dqn_eps, sarsa_eps


# ════════════════════════════════════════════════════════════
# 2. グラフ1: 学習曲線（DQN vs SARSA）+ ε 二軸
# ════════════════════════════════════════════════════════════

def graph1_learning_curve(dqn_scores, sarsa_scores, dqn_eps, sarsa_eps,
                          window=100, save_path=None):
    """
    左軸: スコア移動平均（DQN=青, SARSA=赤）
    右軸: ε（探索率）の減衰曲線
    指摘対応: 縦軸を学習率(ε)の変化にする → 右軸にε追加
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ep = np.arange(1, len(dqn_scores) + 1)

    # ── 生データ（薄く）──
    ax1.plot(ep, dqn_scores,   color="steelblue", alpha=0.15, linewidth=0.6)
    ax1.plot(ep, sarsa_scores, color="tomato",    alpha=0.15, linewidth=0.6)

    # ── 移動平均 ──
    w = min(window, len(dqn_scores))
    if len(dqn_scores) >= w:
        dqn_ma   = np.convolve(dqn_scores,   np.ones(w)/w, mode="valid")
        sarsa_ma = np.convolve(sarsa_scores, np.ones(w)/w, mode="valid")
        x_ma = np.arange(w, len(dqn_scores) + 1)
        ax1.plot(x_ma, dqn_ma,   color="steelblue", linewidth=2.5,
                 label=f"Q学習 / DQN（{w}ep移動平均）")
        ax1.plot(x_ma, sarsa_ma, color="tomato",    linewidth=2.5,
                 label=f"SARSA（{w}ep移動平均）")

    # ── ε（右軸） ──
    ax2.plot(ep, dqn_eps,   color="steelblue", linewidth=1.5, linestyle="--",
             alpha=0.7, label="ε (DQN)")
    ax2.plot(ep, sarsa_eps, color="tomato",    linewidth=1.5, linestyle=":",
             alpha=0.7, label="ε (SARSA)")
    ax2.set_ylabel("探索率 ε（epsilon）", fontsize=13, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.05)

    # ── 装飾 ──
    ax1.set_title("Q学習 vs SARSA  学習曲線", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xlabel("エピソード", fontsize=13)
    ax1.set_ylabel("スコア", fontsize=13)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(bottom=0)

    # 凡例をまとめる
    handles1, labels1 = ax1.get_legend_handles_labels()
    eps_line1 = Line2D([0], [0], color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7)
    eps_line2 = Line2D([0], [0], color="tomato",    linewidth=1.5, linestyle=":",  alpha=0.7)
    handles1 += [eps_line1, eps_line2]
    labels1  += ["ε (Q学習/DQN)", "ε (SARSA)"]
    ax1.legend(handles1, labels1, loc="upper left", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = save_path or os.path.join(SAVE_DIR, "graph1_learning_curve_dqn_vs_sarsa.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")
    return path


# ════════════════════════════════════════════════════════════
# 3. グラフ2: PSO 収束（pbest / gbest）
# ════════════════════════════════════════════════════════════

def run_pso_light(n_particles=6, n_iterations=6, eval_episodes=60, cache_path=None):
    """PSO を軽量設定で実行し、history（pbest/gbest含む）を返す"""
    if cache_path and os.path.exists(cache_path):
        print(f"PSO キャッシュ読み込み: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    def fitness(pos):
        return evaluate(pos, eval_episodes=eval_episodes)

    pso = PSO(n_particles=n_particles, n_iterations=n_iterations,
              w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=42)
    _, history = pso.optimize(fitness_fn=fitness, bounds=BOUNDS,
                              param_names=PARAM_NAMES, verbose=True)

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            # numpy を JSON シリアライズ可能に変換
            safe_hist = []
            for h in history:
                safe_hist.append({
                    "iteration"    : int(h["iteration"]),
                    "gbest_score"  : float(h["gbest_score"]),
                    "gbest_pos"    : [float(v) for v in h["gbest_pos"]],
                    "pbest_scores" : [float(v) for v in h["pbest_scores"]],
                })
            json.dump(safe_hist, f)

    return history


def graph2_pso_convergence(history, save_path=None):
    """
    各粒子のpbest推移（細線）+ 群れのgbest推移（太線）
    指摘対応: 自己ベスト(pbest)とグループのベスト(gbest)を示す
    """
    iters      = [h["iteration"]    for h in history]
    gbest_list = [h["gbest_score"]  for h in history]
    pbest_mat  = np.array([h["pbest_scores"] for h in history])  # shape: (T, N)
    n_particles = pbest_mat.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── 各粒子の pbest（薄い細線）──
    colors_p = plt.cm.Blues(np.linspace(0.3, 0.75, n_particles))
    for i in range(n_particles):
        ax.plot(iters, pbest_mat[:, i], color=colors_p[i],
                linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"粒子{i+1} pbest" if i == 0 else None)

    # 凡例用に1本まとめる
    pbest_proxy = Line2D([0], [0], color="steelblue", linewidth=1.5, linestyle="--",
                         alpha=0.7, label="各粒子の自己ベスト（pbest）")

    # ── 群れの gbest（太い赤線）──
    ax.plot(iters, gbest_list, color="crimson", linewidth=3.0, marker="o",
            markersize=8, zorder=5, label="群れの最良値（gbest）")

    # ── pbest の帯（min〜max）──
    pbest_min = pbest_mat.min(axis=1)
    pbest_max = pbest_mat.max(axis=1)
    ax.fill_between(iters, pbest_min, pbest_max, color="steelblue", alpha=0.1,
                    label="pbest の分布範囲")

    # ── スコア注釈 ──
    for i, (it, score) in enumerate(zip(iters, gbest_list)):
        if i == 0 or score > gbest_list[i-1]:
            ax.annotate(f"{score:.0f}", xy=(it, score),
                        xytext=(5, 8), textcoords="offset points",
                        fontsize=10, color="crimson", fontweight="bold")

    ax.set_title("PSO 収束グラフ  ―  pbest（自己ベスト）vs gbest（群れベスト）",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("世代（Iteration）", fontsize=13)
    ax.set_ylabel("スコア（適応度）", fontsize=13)
    ax.set_xticks(iters)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    handles = [pbest_proxy] + [h for h, l in zip(handles, labels)
                               if "群れ" in l or "分布" in l]
    labels  = ["各粒子の自己ベスト（pbest）",
               "群れの最良値（gbest）",
               "pbest の分布範囲"]
    ax.legend(handles, labels, fontsize=11, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    path = save_path or os.path.join(SAVE_DIR, "graph2_pso_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")
    return path


# ════════════════════════════════════════════════════════════
# 4. グラフ3: PSO パラメータ軌跡
# ════════════════════════════════════════════════════════════

def graph3_pso_params(history, save_path=None):
    """PSO 最適化中の gbest パラメータの推移を4パネルで表示"""
    iters = [h["iteration"] for h in history]

    def decode_all(history):
        lrs, eps_ends, eps_decays, gammas = [], [], [], []
        for h in history:
            p = decode_params(h["gbest_pos"])
            lrs.append(p["lr"])
            eps_ends.append(p["epsilon_end"])
            eps_decays.append(p["epsilon_decay"])
            gammas.append(p["gamma"])
        return lrs, eps_ends, eps_decays, gammas

    lrs, eps_ends, eps_decays, gammas = decode_all(history)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PSO ハイパーパラメータ最適化  ―  gbest パラメータ軌跡",
                 fontsize=14, fontweight="bold")

    datasets = [
        (axes[0,0], lrs,        "学習率 (Learning Rate)",   "tomato",       True),
        (axes[0,1], eps_ends,   "探索率下限 (ε_end)",       "darkorange",   False),
        (axes[1,0], eps_decays, "探索率減衰 (ε_decay)",     "mediumseagreen", False),
        (axes[1,1], gammas,     "割引率 (γ / gamma)",       "mediumpurple", False),
    ]

    for ax, data, title, color, log_scale in datasets:
        ax.plot(iters, data, "o-", color=color, linewidth=2.2, markersize=7)
        ax.fill_between(iters, data, min(data), color=color, alpha=0.1)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("世代", fontsize=11)
        ax.grid(True, alpha=0.25, which="both" if log_scale else "major")
        ax.set_xticks(iters)
        # 最終値注釈
        ax.annotate(f"最終: {data[-1]:.5g}", xy=(iters[-1], data[-1]),
                    xytext=(-50, 8), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    plt.tight_layout()
    path = save_path or os.path.join(SAVE_DIR, "graph3_pso_param_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")
    return path


# ════════════════════════════════════════════════════════════
# 5. グラフ4: DQN vs SARSA 統計比較
# ════════════════════════════════════════════════════════════

def graph4_statistics(dqn_scores, sarsa_scores, save_path=None):
    """平均・中央値・最高・後半平均のグループ棒グラフ"""
    last = min(100, len(dqn_scores))
    metrics = ["全体平均", "中央値", "最高スコア", f"後半{last}ep\n平均"]
    dqn_vals = [
        np.mean(dqn_scores),
        np.median(dqn_scores),
        np.max(dqn_scores),
        np.mean(dqn_scores[-last:]),
    ]
    sarsa_vals = [
        np.mean(sarsa_scores),
        np.median(sarsa_scores),
        np.max(sarsa_scores),
        np.mean(sarsa_scores[-last:]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Q学習 vs SARSA  ―  スコア統計比較", fontsize=14, fontweight="bold")

    # ── 左: 棒グラフ ──
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, dqn_vals,   w, label="Q学習 / DQN",
                color="steelblue", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, sarsa_vals, w, label="SARSA",
                color="tomato",    alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(max(dqn_vals), max(sarsa_vals)) * 0.01,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("スコア", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(max(dqn_vals), max(sarsa_vals)) * 1.18)

    # 勝者ハイライト
    for i, (dv, sv) in enumerate(zip(dqn_vals, sarsa_vals)):
        winner_x = (x[i] - w/2) if dv >= sv else (x[i] + w/2)
        ax.annotate("★", xy=(winner_x, max(dv, sv) * 1.05),
                    ha="center", fontsize=12, color="gold")

    # ── 右: スコア分布 ──
    ax = axes[1]
    ax.hist(dqn_scores,   bins=25, color="steelblue", alpha=0.6,
            density=True, label="Q学習 / DQN", orientation="vertical")
    ax.hist(sarsa_scores, bins=25, color="tomato",    alpha=0.6,
            density=True, label="SARSA", orientation="vertical")
    ax.set_title("スコア分布", fontsize=12)
    ax.set_xlabel("スコア", fontsize=11)
    ax.set_ylabel("密度", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = save_path or os.path.join(SAVE_DIR, "graph4_final_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")
    return path


# ════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════

def main(args):
    os.chdir(SAVE_DIR)

    # PSO最適パラメータ（前回の tuning 結果）
    lr            = 0.002056
    epsilon_decay = 0.99454
    gamma         = 0.9294
    epsilon_end   = 0.0248

    print("\n" + "="*60)
    print("  グラフ自動生成スクリプト")
    print(f"  訓練エピソード数: {args.episodes}")
    print(f"  PSO: 粒子{args.pso_particles}×世代{args.pso_iterations}")
    print("="*60)

    # ── 1. DQN / SARSA 訓練 ──
    cache = "scores_cache.json"
    if args.skip_train:
        cache = None
    dqn_scores, sarsa_scores, dqn_eps, sarsa_eps = run_training(
        args.episodes, lr, epsilon_decay, gamma, epsilon_end, cache)

    # ── 2. PSO 実行 ──
    pso_cache = "pso_history_cache.json"
    if args.skip_pso:
        pso_cache = None
    history = run_pso_light(
        n_particles   = args.pso_particles,
        n_iterations  = args.pso_iterations,
        eval_episodes = args.pso_eval_episodes,
        cache_path    = pso_cache,
    )

    # ── 3. グラフ生成 ──
    print("\nグラフ生成中...")
    window = min(100, args.episodes // 5)

    p1 = graph1_learning_curve(dqn_scores, sarsa_scores, dqn_eps, sarsa_eps, window)
    p2 = graph2_pso_convergence(history)
    p3 = graph3_pso_params(history)
    p4 = graph4_statistics(dqn_scores, sarsa_scores)

    # ── 4. サマリー ──
    last = min(100, len(dqn_scores))
    print("\n" + "="*60)
    print(f"  {'指標':<18} {'Q学習/DQN':>10} {'SARSA':>10}")
    print("-"*60)
    rows = [
        ("全体平均",        np.mean(dqn_scores),         np.mean(sarsa_scores)),
        ("中央値",          np.median(dqn_scores),       np.median(sarsa_scores)),
        ("最高スコア",      np.max(dqn_scores),          np.max(sarsa_scores)),
        (f"後半{last}ep平均", np.mean(dqn_scores[-last:]), np.mean(sarsa_scores[-last:])),
    ]
    for label, dv, sv in rows:
        w = "<-" if dv > sv else ("->" if sv > dv else "  ")
        print(f"  {label:<18} {dv:>10.1f} {sv:>10.1f}  {w}")
    print("="*60)

    print("\n生成ファイル一覧:")
    for p in [p1, p2, p3, p4]:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="グラフ自動生成")
    parser.add_argument("--episodes",          type=int, default=300,
                        help="DQN/SARSA 訓練エピソード数")
    parser.add_argument("--pso-particles",     type=int, default=5,
                        help="PSO 粒子数")
    parser.add_argument("--pso-iterations",    type=int, default=5,
                        help="PSO 世代数")
    parser.add_argument("--pso-eval-episodes", type=int, default=60,
                        help="PSO 評価エピソード数（1粒子あたり）")
    parser.add_argument("--skip-train",        action="store_true",
                        help="DQN/SARSA 訓練をスキップ（キャッシュ使用）")
    parser.add_argument("--skip-pso",          action="store_true",
                        help="PSO 実行をスキップ（キャッシュ使用）")
    args = parser.parse_args()
    main(args)
