"""
PSOによるDQNハイパーパラメータ自動最適化
================================================================
使い方:
  python pso_tuning.py                    # デフォルト設定で実行
  python pso_tuning.py --particles 15     # 粒子数を増やす
  python pso_tuning.py --iterations 15   # 世代数を増やす
  python pso_tuning.py --eval-episodes 150  # 評価エピソード数
================================================================
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from pso_optimizer import PSO
from plot_curve import plot_learning_curve


# ──────────────────────────────────────────────
# 探索するハイパーパラメータの定義
# ──────────────────────────────────────────────
# 学習率 (log10スケール): 10^[-4, -2] = [0.0001, 0.01]
# epsilon_end           : [0.01, 0.20]
# epsilon_decay         : [0.990, 0.9995]
# gamma (割引率)        : [0.90, 0.999]

PARAM_NAMES = ["log_lr", "epsilon_end", "epsilon_decay", "gamma"]

BOUNDS = [
    (-4.0, -2.0),      # log10(learning_rate)
    (0.01,  0.20),     # epsilon_end
    (0.990, 0.9995),   # epsilon_decay
    (0.90,  0.999),    # gamma
]


def decode_params(pos):
    """粒子の位置ベクトル → 実際のハイパーパラメータ dict"""
    return {
        "lr"           : float(10 ** pos[0]),
        "epsilon_end"  : float(pos[1]),
        "epsilon_decay": float(pos[2]),
        "gamma"        : float(pos[3]),
    }


def evaluate(pos, eval_episodes=100, max_steps=800):
    """
    粒子の位置を受け取り、その超パラメータでDQNを訓練して
    最後50エピソードの平均スコアを返す（適応度関数）
    """
    params = decode_params(pos)
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]

    agent = DQNAgent(
        obs_size      = obs_size,
        n_actions     = 4,
        lr            = params["lr"],
        gamma         = params["gamma"],
        epsilon_start = 1.0,
        epsilon_end   = params["epsilon_end"],
        epsilon_decay = params["epsilon_decay"],
        batch_size    = 64,
        target_update_freq = 200,
    )

    scores = []
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        steps = 0
        while steps < max_steps:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.train_step()
            obs = next_obs
            steps += 1
            if terminated or truncated:
                break
        scores.append(info["score"])

    env.close()
    # 後半50エピソードの平均を適応度とする
    return float(np.mean(scores[-50:]))


def plot_pso_history(history, save_path="pso_convergence.png"):
    """PSO収束グラフを保存"""
    iters  = [h["iteration"]   for h in history]
    scores = [h["gbest_score"] for h in history]

    # 各世代のgbest_posを展開
    params_over_time = {name: [] for name in PARAM_NAMES}
    real_params = {
        "lr"           : [],
        "epsilon_end"  : [],
        "epsilon_decay": [],
        "gamma"        : [],
    }
    for h in history:
        p = decode_params(h["gbest_pos"])
        params_over_time["log_lr"].append(h["gbest_pos"][0])
        params_over_time["epsilon_end"].append(h["gbest_pos"][1])
        params_over_time["epsilon_decay"].append(h["gbest_pos"][2])
        params_over_time["gamma"].append(h["gbest_pos"][3])
        for k, v in p.items():
            real_params[k].append(v)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PSO Convergence - DQN Hyperparameter Optimization", fontsize=14)

    # ① 収束曲線
    ax = axes[0, 0]
    ax.plot(iters, scores, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.fill_between(iters, scores, min(scores), alpha=0.15, color="steelblue")
    ax.set_title("Best Score (Fitness)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Score (last 50 ep)")
    ax.grid(True, alpha=0.3)

    # ② 学習率
    ax = axes[0, 1]
    ax.plot(iters, real_params["lr"], "s-", color="tomato", linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("lr (log scale)")
    ax.grid(True, alpha=0.3, which="both")

    # ③ epsilon_end
    ax = axes[0, 2]
    ax.plot(iters, real_params["epsilon_end"], "^-", color="darkorange", linewidth=2)
    ax.set_title("Epsilon End")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("epsilon_end")
    ax.grid(True, alpha=0.3)

    # ④ epsilon_decay
    ax = axes[1, 0]
    ax.plot(iters, real_params["epsilon_decay"], "D-", color="mediumseagreen", linewidth=2)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("epsilon_decay")
    ax.grid(True, alpha=0.3)

    # ⑤ gamma
    ax = axes[1, 1]
    ax.plot(iters, real_params["gamma"], "P-", color="mediumpurple", linewidth=2)
    ax.set_title("Gamma (Discount Rate)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("gamma")
    ax.grid(True, alpha=0.3)

    # ⑥ パラメータ最終値のサマリー
    ax = axes[1, 2]
    ax.axis("off")
    final = decode_params(history[-1]["gbest_pos"])
    summary = (
        f"== 最適化結果 ==\n\n"
        f"最良スコア:     {history[-1]['gbest_score']:.1f}\n\n"
        f"learning_rate:  {final['lr']:.6f}\n"
        f"epsilon_end:    {final['epsilon_end']:.4f}\n"
        f"epsilon_decay:  {final['epsilon_decay']:.5f}\n"
        f"gamma:          {final['gamma']:.4f}"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=12, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PSO収束グラフ保存: {save_path}")


def retrain_best(best_params, episodes=800):
    """最良パラメータで本格訓練し、学習曲線を保存"""
    print("\n" + "="*60)
    print("  最良パラメータで本格訓練開始")
    print(f"  learning_rate : {best_params['lr']:.6f}")
    print(f"  epsilon_end   : {best_params['epsilon_end']:.4f}")
    print(f"  epsilon_decay : {best_params['epsilon_decay']:.5f}")
    print(f"  gamma         : {best_params['gamma']:.4f}")
    print("="*60 + "\n")

    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]

    agent = DQNAgent(
        obs_size      = obs_size,
        n_actions     = 4,
        lr            = best_params["lr"],
        gamma         = best_params["gamma"],
        epsilon_start = 1.0,
        epsilon_end   = best_params["epsilon_end"],
        epsilon_decay = best_params["epsilon_decay"],
        batch_size    = 128,
        target_update_freq = 300,
    )

    scores = []
    best_score = 0
    t0 = time.time()

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
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save("pacman_model_pso.pt")

        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            bar = "#" * int(avg / 20)
            print(f"  Ep {ep:3d}/{episodes}  Avg100={avg:5.0f} {bar:<25}  Best={best_score}  e={agent.epsilon:.3f}")

    env.close()
    elapsed = time.time() - t0
    print(f"\n  訓練完了 ({elapsed:.1f}秒)  最高スコア: {best_score}")
    print(f"  モデル保存: pacman_model_pso.pt")

    plot_learning_curve(scores,
                        title="AI Learning Curve (PSO-tuned DQN)",
                        save_path="learning_curve_pso.png",
                        color="darkorange")
    return agent, scores


def main(args):
    print("\n" + "="*60)
    print("  PSO × DQN ハイパーパラメータ最適化")
    print("="*60)
    print("\n【PSO設定】")
    print(f"  粒子数     : {args.particles}")
    print(f"  世代数     : {args.iterations}")
    print(f"  評価ep数   : {args.eval_episodes}")
    print(f"  慣性重み   : 0.9 → 0.4（線形減衰）")
    print(f"  認知係数c1 : 1.5")
    print(f"  社会係数c2 : 1.5")

    # 評価関数（粒子位置 → 平均スコア）
    def fitness(pos):
        return evaluate(pos, eval_episodes=args.eval_episodes)

    # PSO実行
    pso = PSO(
        n_particles  = args.particles,
        n_iterations = args.iterations,
        w_start=0.9, w_end=0.4,
        c1=1.5, c2=1.5,
        seed=42,
    )
    best_pos_dict, history = pso.optimize(
        fitness_fn   = fitness,
        bounds       = BOUNDS,
        param_names  = PARAM_NAMES,
        verbose      = True,
    )

    # decode して実際の値に変換
    best_params = decode_params(pso.gbest_pos)

    # PSO収束グラフ保存
    plot_pso_history(history, save_path="pso_convergence.png")

    # 最良パラメータで本格訓練
    if not args.search_only:
        retrain_best(best_params, episodes=args.train_episodes)
    else:
        print("\n[--search-only モード: 本格訓練をスキップ]")
        print("以下のコマンドで本格訓練できます:")
        print(f"  python train.py --lr {best_params['lr']:.6f} ...")

    print("\n生成されたファイル:")
    print("  pso_convergence.png    : PSO収束グラフ")
    if not args.search_only:
        print("  learning_curve_pso.png : PSO最適パラメータでの学習曲線")
        print("  pacman_model_pso.pt    : PSO最適化済みモデル")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO によるDQNハイパーパラメータ最適化")
    parser.add_argument("--particles",      type=int, default=8,   help="粒子数")
    parser.add_argument("--iterations",     type=int, default=8,   help="世代数")
    parser.add_argument("--eval-episodes",  type=int, default=100, help="1粒子あたりの評価エピソード数")
    parser.add_argument("--train-episodes", type=int, default=800, help="最良パラメータでの本格訓練エピソード数")
    parser.add_argument("--search-only",    action="store_true",   help="探索のみ（本格訓練をスキップ）")
    args = parser.parse_args()
    main(args)
