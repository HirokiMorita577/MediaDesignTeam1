"""
クイックデモ: 100エピソード高速訓練 → カーナビ表示でプレイ観戦
python quick_demo.py
"""
import numpy as np
import time
from pacman_env import PacManEnv
from dqn_agent import DQNAgent


def main():
    print("=" * 60)
    print("  Pac-Man RL カーナビ風経路表示 クイックデモ")
    print("=" * 60)

    # ── 1. 高速訓練 (描画なし) ──
    print("\n[1/2] 200エピソード高速訓練中...")
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    agent = DQNAgent(obs_size=obs_size, n_actions=4,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99)

    scores = []
    t0 = time.time()
    for ep in range(1, 201):
        obs, _ = env.reset()
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.train_step()
            obs = next_obs
            if terminated or truncated:
                break
        scores.append(info["score"])
        if ep % 50 == 0:
            print(f"  Ep {ep:3d}/200  Avg50={np.mean(scores[-50:]):.0f}  ε={agent.epsilon:.3f}")

    agent.save("pacman_model.pt")
    print(f"  訓練完了 ({time.time()-t0:.1f}秒)  最高スコア: {max(scores)}")
    env.close()

    # ── 2. カーナビ風表示でプレイ ──
    print("\n[2/2] カーナビ風経路表示でプレイ観戦 (3ゲーム)")
    print("  青→緑グラデーション: A*計画経路")
    print("  赤丸: 目標エサ")
    print("  ESCキーで終了\n")

    import pygame
    env2 = PacManEnv(render_mode="human")
    agent.epsilon = 0.05  # ほぼグリーディ

    for ep in range(3):
        obs, _ = env2.reset()
        env2.render()
        print(f"Episode {ep+1}/3 開始")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env2.close()
                    return
            action = agent.select_action(obs, training=False)
            obs, _, terminated, truncated, info = env2.step(action)
            if terminated or truncated:
                print(f"  スコア: {info['score']}")
                pygame.time.wait(1000)
                break

    env2.close()
    print("\nデモ完了!")


if __name__ == "__main__":
    main()
