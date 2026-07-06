"""
DQN vs SARSA 比較スクリプト
================================================================
使い方:
  python compare.py                    # デフォルト(500ep)
  python compare.py --episodes 1000    # エピソード数指定
  python compare.py --play             # 比較後にDQN/SARSAを交互に観戦
================================================================
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from sarsa_agent import SARSAAgent


def _get_jp_font():
    candidates = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAGothic", "Noto Sans CJK JP"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


# ──────────────────────────────────────────────────────────
# 訓練ループ
# ──────────────────────────────────────────────────────────

def train_dqn(episodes, lr, epsilon_decay, gamma, epsilon_end):
    """DQN訓練ループ"""
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    agent = DQNAgent(
        obs_size=obs_size, n_actions=4,
        lr=lr, gamma=gamma,
        epsilon_start=1.0, epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=128, target_update_freq=300,
    )

    scores, best = [], 0
    t0 = time.time()
    print(f"\n[DQN] 訓練開始  {episodes}ep")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.train_step()
            obs = next_obs
            if terminated or truncated:
                break
        scores.append(info["score"])
        if info["score"] > best:
            best = info["score"]
            agent.save("pacman_model_dqn_cmp.pt")
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            bar = "#" * int(avg / 20)
            print(f"  [DQN]  Ep {ep:4d}/{episodes}  Avg100={avg:5.0f} {bar:<25}  Best={best}  e={agent.epsilon:.3f}")

    env.close()
    print(f"  [DQN]  完了 ({time.time()-t0:.1f}s)  Best={best}")
    return scores, agent


def train_sarsa(episodes, lr, epsilon_decay, gamma, epsilon_end):
    """SARSA訓練ループ (s,a,r,s',a' のタプルを使用)"""
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    agent = SARSAAgent(
        obs_size=obs_size, n_actions=4,
        lr=lr, gamma=gamma,
        epsilon_start=1.0, epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=128,
    )

    scores, best = [], 0
    t0 = time.time()
    print(f"\n[SARSA] 訓練開始  {episodes}ep")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        action = agent.select_action(obs, training=True)   # a0を先に選ぶ

        while True:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # 次の行動を現在の方策で選ぶ (SARSAの核心)
            next_action = agent.select_action(next_obs, training=True)
            agent.push(obs, action, reward, next_obs, next_action, float(done))
            agent.train_step()
            obs    = next_obs
            action = next_action   # 選んだ行動を実際に使う
            if done:
                break

        scores.append(info["score"])
        if info["score"] > best:
            best = info["score"]
            agent.save("pacman_model_sarsa_cmp.pt")
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            bar = "#" * int(avg / 20)
            print(f"  [SARSA] Ep {ep:4d}/{episodes}  Avg100={avg:5.0f} {bar:<25}  Best={best}  e={agent.epsilon:.3f}")

    env.close()
    print(f"  [SARSA] 完了 ({time.time()-t0:.1f}s)  Best={best}")
    return scores, agent


# ──────────────────────────────────────────────────────────
# 比較グラフ
# ──────────────────────────────────────────────────────────

def plot_comparison(dqn_scores, sarsa_scores, window=100,
                    save_path="comparison.png"):
    jp_font = _get_jp_font()
    if jp_font:
        plt.rcParams["font.family"] = jp_font

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DQN vs SARSA Comparison", fontsize=15, fontweight="bold")

    episodes = np.arange(1, len(dqn_scores) + 1)

    # ── 左: 学習曲線（両アルゴリズム重ね描き）──
    ax = axes[0]
    ax.plot(episodes, dqn_scores,   color="steelblue",  alpha=0.2, linewidth=0.5)
    ax.plot(episodes, sarsa_scores, color="tomato",     alpha=0.2, linewidth=0.5)

    if len(dqn_scores) >= window:
        dqn_ma   = np.convolve(dqn_scores,   np.ones(window)/window, mode="valid")
        sarsa_ma = np.convolve(sarsa_scores, np.ones(window)/window, mode="valid")
        x_ma = np.arange(window, len(dqn_scores) + 1)
        ax.plot(x_ma, dqn_ma,   color="steelblue", linewidth=2.2, label=f"DQN ({window}ep MA)")
        ax.plot(x_ma, sarsa_ma, color="tomato",    linewidth=2.2, label=f"SARSA ({window}ep MA)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # ── 中央: スコア分布（ヒストグラム）──
    ax = axes[1]
    ax.hist(dqn_scores,   bins=30, color="steelblue", alpha=0.6, label="DQN",   density=True)
    ax.hist(sarsa_scores, bins=30, color="tomato",    alpha=0.6, label="SARSA", density=True)
    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # ── 右: 統計比較（棒グラフ）──
    ax = axes[2]
    metrics = ["Mean", "Median", "Max", "Last100\nMean"]
    last = min(100, len(dqn_scores))
    dqn_vals   = [
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
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, dqn_vals,   w, label="DQN",   color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w/2, sarsa_vals, w, label="SARSA", color="tomato",    alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Statistics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"比較グラフ保存: {save_path}")


def print_summary(dqn_scores, sarsa_scores):
    last = min(100, len(dqn_scores))
    print("\n" + "="*55)
    print(f"  {'指標':<20}  {'DQN':>10}  {'SARSA':>10}")
    print("-"*55)
    rows = [
        ("平均スコア",        np.mean(dqn_scores),           np.mean(sarsa_scores)),
        ("中央値",            np.median(dqn_scores),         np.median(sarsa_scores)),
        ("最高スコア",        np.max(dqn_scores),            np.max(sarsa_scores)),
        ("最低スコア",        np.min(dqn_scores),            np.min(sarsa_scores)),
        ("標準偏差",          np.std(dqn_scores),            np.std(sarsa_scores)),
        (f"後半{last}ep平均", np.mean(dqn_scores[-last:]),   np.mean(sarsa_scores[-last:])),
    ]
    for label, dv, sv in rows:
        winner = "<-" if dv > sv else ("->" if sv > dv else "  ")
        print(f"  {label:<20}  {dv:>10.1f}  {sv:>10.1f}  {winner}")
    print("="*55)

    winner = "DQN" if np.mean(dqn_scores[-last:]) > np.mean(sarsa_scores[-last:]) else "SARSA"
    print(f"  後半{last}ep平均スコアの勝者: {winner}")
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────
# 観戦モード
# ──────────────────────────────────────────────────────────

def play_comparison(dqn_agent, sarsa_agent, episodes_each=2):
    import pygame
    from pacman_env import PacManEnv

    for name, agent, model in [("DQN", dqn_agent, None), ("SARSA", sarsa_agent, None)]:
        env = PacManEnv(render_mode="human")
        print(f"\n--- {name} プレイ観戦 ---")

        for ep in range(episodes_each):
            obs, _ = env.reset()
            env.render()

            # Q値をセット
            qv = agent.get_qvalues(obs)
            env.set_qvalues(qv)

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        env.close()
                        return

                action = agent.select_action(obs, training=False)
                env.set_qvalues(agent.get_qvalues(obs))
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"  {name} Ep{ep+1}: score={info['score']}")
                    pygame.time.wait(1200)
                    break

        env.close()


# ──────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────

def main(args):
    # PSO最適パラメータがあれば使用、なければデフォルト
    lr            = args.lr
    epsilon_decay = args.epsilon_decay
    gamma         = args.gamma
    epsilon_end   = args.epsilon_end

    print("\n" + "="*55)
    print("  DQN vs SARSA 比較実験")
    print(f"  エピソード数: {args.episodes}")
    print(f"  lr={lr}  decay={epsilon_decay}  gamma={gamma}  e_end={epsilon_end}")
    print("="*55)

    # 訓練
    dqn_scores,   dqn_agent   = train_dqn(  args.episodes, lr, epsilon_decay, gamma, epsilon_end)
    sarsa_scores, sarsa_agent = train_sarsa(args.episodes, lr, epsilon_decay, gamma, epsilon_end)

    # 結果出力
    print_summary(dqn_scores, sarsa_scores)

    # 比較グラフ保存
    plot_comparison(dqn_scores, sarsa_scores,
                    window=min(100, args.episodes // 5),
                    save_path="comparison.png")

    print("生成ファイル:")
    print("  comparison.png            : DQN vs SARSA 比較グラフ")
    print("  pacman_model_dqn_cmp.pt   : DQN最良モデル")
    print("  pacman_model_sarsa_cmp.pt : SARSA最良モデル")

    # 観戦
    if args.play:
        play_comparison(dqn_agent, sarsa_agent, episodes_each=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN vs SARSA 比較")
    parser.add_argument("--episodes",      type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=0.002056, help="学習率（PSOデフォルト）")
    parser.add_argument("--epsilon-decay", type=float, default=0.99454,  help="epsilon減衰率")
    parser.add_argument("--gamma",         type=float, default=0.9294,   help="割引率")
    parser.add_argument("--epsilon-end",   type=float, default=0.0248,   help="最終epsilon")
    parser.add_argument("--play",          action="store_true",           help="訓練後に観戦")
    args = parser.parse_args()
    main(args)
