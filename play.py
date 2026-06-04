"""
カーナビ風経路表示でAIプレイを観戦
使い方:
  python play.py                    # 訓練済みモデルで自動プレイ
  python play.py --human            # 手動プレイ (矢印キー)
  python play.py --model none       # 未訓練エージェントでランダムプレイ
  python play.py --episodes 5       # 5ゲーム連続プレイ
"""
import argparse
import os
import sys
import pygame
import numpy as np
from pacman_env import PacManEnv
from dqn_agent import DQNAgent


ACTION_MAP = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_w: 0,
    pygame.K_s: 1,
    pygame.K_a: 2,
    pygame.K_d: 3,
}


def play(args):
    env = PacManEnv(render_mode="human")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = None
    if not args.human:
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions,
                         epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0)
        model_path = args.model if args.model != "none" else "pacman_model.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"モデル読込: {model_path}")
        else:
            print(f"[警告] モデルファイル '{model_path}' が見つかりません。ランダムエージェントで動作します。")
            agent.epsilon = 1.0

    total_scores = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        env.render()
        total_reward = 0
        action = 3  # 初期方向: 右

        print(f"\n=== Episode {ep+1}/{args.episodes} ===")
        print("カーナビ風経路表示 ON | 青→緑グラデーションが計画経路、赤丸が目標エサ")

        running = True
        while running:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    if args.human and event.key in ACTION_MAP:
                        action = ACTION_MAP[event.key]

            # AIアクション選択
            if not args.human and agent:
                action = agent.select_action(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                running = False

        score = info["score"]
        total_scores.append(score)
        print(f"スコア: {score}  累積報酬: {total_reward:.1f}")

        # ゲームオーバー画面を少し表示
        if env.screen:
            pygame.time.wait(1500)

    if total_scores:
        print(f"\n{'='*40}")
        print(f"全{args.episodes}ゲーム完了")
        print(f"平均スコア: {np.mean(total_scores):.1f}")
        print(f"最高スコア: {max(total_scores)}")
        print(f"最低スコア: {min(total_scores)}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--human", action="store_true", help="手動プレイモード")
    parser.add_argument("--model", type=str, default="pacman_model.pt")
    args = parser.parse_args()
    play(args)
