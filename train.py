"""
訓練スクリプト
使い方:
  python train.py              # 訓練開始
  python train.py --resume     # 前回の続きから
  python train.py --episodes 500
"""
import argparse
import os
import time
import numpy as np
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from plot_curve import plot_learning_curve


def train(args):
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_size=obs_size,
        n_actions=n_actions,
        lr=1e-4,
        epsilon_start=1.0 if not args.resume else 0.3,
        epsilon_end=0.05,
        epsilon_decay=0.997,
    )

    model_path = "pacman_model.pt"
    if args.resume and os.path.exists(model_path):
        agent.load(model_path)

    scores = []
    losses = []
    best_score = -float("inf")
    start_time = time.time()

    print(f"\n{'='*60}")
    print("  Pac-Man DQN 訓練開始")
    print(f"  エピソード数: {args.episodes}")
    print(f"  観測次元: {obs_size}  行動数: {n_actions}")
    print(f"{'='*60}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        ep_losses = []

        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.push(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            total_reward += reward

            if done:
                break

        scores.append(info["score"])
        if ep_losses:
            losses.append(np.mean(ep_losses))

        # ログ出力
        if ep % args.log_interval == 0:
            avg_score = np.mean(scores[-50:])
            avg_loss  = np.mean(losses[-50:]) if losses else 0
            elapsed   = time.time() - start_time
            bar_len   = int(avg_score / 50)
            bar       = "#" * min(bar_len, 30)
            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"Score {info['score']:5d} | Avg50 {avg_score:6.1f} {bar:<30} | "
                f"Loss {avg_loss:.4f} | e={agent.epsilon:.3f} | {elapsed:.0f}s"
            )

        # ベストモデル保存
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(model_path)

        # 定期保存 + 途中グラフ
        if ep % 200 == 0:
            agent.save(f"pacman_model_ep{ep}.pt")
            plot_learning_curve(scores, save_path="learning_curve.png")

    env.close()
    print(f"\n訓練完了! ベストスコア: {best_score}")
    print(f"モデル保存済み: {model_path}")

    # 最終グラフ保存
    plot_learning_curve(scores, save_path="learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    train(args)
