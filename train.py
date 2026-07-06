"""
訓練スクリプト（じっくり長期訓練用）
使い方:
  python train.py                      # 20000ep 新規訓練
  python train.py --episodes 50000     # エピソード数指定
  python train.py --resume             # 前回の続きから
  python train.py --resume --episodes 10000  # 続きから追加10000ep
"""
import argparse
import csv
import os
import time
import numpy as np
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from plot_curve import plot_learning_curve

# PSO最適化済みパラメータ
DEFAULT_LR            = 0.002056
DEFAULT_EPSILON_END   = 0.0248
DEFAULT_EPSILON_DECAY = 0.99454
DEFAULT_GAMMA         = 0.9294


def train(args):
    env = PacManEnv(render_mode=None, ghost_mode=args.ghost_mode)
    obs_size  = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_size=obs_size,
        n_actions=n_actions,
        lr            = args.lr,
        gamma         = args.gamma,
        epsilon_start = 1.0 if not args.resume else args.epsilon_end,
        epsilon_end   = args.epsilon_end,
        epsilon_decay = args.epsilon_decay,
        batch_size    = 128,
        target_update_freq = 300,
    )

    model_path = args.model
    if args.resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"前回モデル読込: {model_path}")
    elif args.resume:
        print(f"[警告] {model_path} が見つかりません。新規訓練を開始します。")

    scores, losses = [], []
    best_score  = 0
    start_time  = time.time()
    save_interval = max(500, args.episodes // 20)   # 5%ごとに中間保存

    # CSVログのパスとヘッダー
    csv_path = os.path.splitext(model_path)[0] + "_log.csv"
    csv_is_new = not os.path.exists(csv_path) or not args.resume
    csv_file = open(csv_path, "w" if csv_is_new else "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["episode", "score", "epsilon",
                             "ghost_mode", "lr", "gamma", "epsilon_decay"])

    print(f"\n{'='*60}")
    print(f"  Pac-Man DQN 長期訓練")
    print(f"  エピソード数  : {args.episodes}")
    print(f"  学習率        : {args.lr}")
    print(f"  gamma         : {args.gamma}")
    print(f"  epsilon_decay : {args.epsilon_decay}")
    print(f"  epsilon_end   : {args.epsilon_end}")
    print(f"  ゴーストAI    : {args.ghost_mode}")
    print(f"  モデル保存先  : {model_path}")
    print(f"{'='*60}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
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
            if done:
                break

        scores.append(info["score"])
        if ep_losses:
            losses.append(np.mean(ep_losses))

        # CSVに1行書き込み
        csv_writer.writerow([ep, info["score"], f"{agent.epsilon:.6f}",
                             args.ghost_mode, args.lr, args.gamma, args.epsilon_decay])

        # ベストモデル保存
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(model_path)

        # ログ出力
        if ep % args.log_interval == 0:
            avg   = np.mean(scores[-100:])
            loss_ = np.mean(losses[-100:]) if losses else 0
            bar   = "#" * min(int(avg / 30), 25)
            elapsed = time.time() - start_time
            h, m = divmod(int(elapsed), 3600)
            m, s = divmod(m, 60)
            print(
                f"Ep {ep:6d}/{args.episodes} | "
                f"Avg100={avg:6.0f} {bar:<25} | "
                f"Best={best_score:5d} | "
                f"e={agent.epsilon:.4f} | "
                f"{h:02d}:{m:02d}:{s:02d}"
            )

        # 定期中間保存 + グラフ更新
        if ep % save_interval == 0:
            mid_path = f"pacman_model_ep{ep}.pt"
            agent.save(mid_path)
            plot_learning_curve(scores, save_path="learning_curve.png")
            print(f"  [中間保存] {mid_path}  learning_curve.png 更新")

    env.close()
    csv_file.close()
    print(f"ログ保存       : {csv_path}")

    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n訓練完了!  {h:02d}:{m:02d}:{s:02d}")
    print(f"ベストスコア : {best_score}")
    print(f"モデル保存   : {model_path}")

    plot_learning_curve(scores, save_path="learning_curve.png")
    print("学習曲線保存 : learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man DQN 長期訓練")
    parser.add_argument("--episodes",       type=int,   default=20000)
    parser.add_argument("--resume",         action="store_true", help="前回の続きから")
    parser.add_argument("--model",          type=str,   default="pacman_model.pt")
    parser.add_argument("--log-interval",   type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma",          type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon-decay",  type=float, default=DEFAULT_EPSILON_DECAY)
    parser.add_argument("--epsilon-end",    type=float, default=DEFAULT_EPSILON_END)
    parser.add_argument("--ghost-mode",     type=str,   default="classic",
                        choices=["classic", "bfs"],
                        help="ゴーストAI: classic=本家4体行動(デフォルト), bfs=全員直接追跡")
    args = parser.parse_args()
    train(args)
