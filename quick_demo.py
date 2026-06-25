"""
クイックデモ: 100エピソード高速訓練 → カーナビ表示でプレイ観戦
python quick_demo.py
"""
import numpy as np
import time
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from plot_curve import plot_learning_curve


def main():
    print("=" * 60)
    print("  Pac-Man RL カーナビ風経路表示 クイックデモ")
    print("=" * 60)

    # ── 1. 高速訓練 (描画なし) ──
    EPISODES = 800
    print(f"\n[1/2] {EPISODES}エピソード高速訓練中...")
    env = PacManEnv(render_mode=None)
    obs_size = env.observation_space.shape[0]
    agent = DQNAgent(obs_size=obs_size, n_actions=4,
                     epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                     batch_size=128, target_update_freq=300)

    scores = []
    best_score = 0
    t0 = time.time()
    for ep in range(1, EPISODES + 1):
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
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save("pacman_model.pt")  # ベストスコア時のみ保存
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            bar = "#" * int(avg / 20)
            print(f"  Ep {ep:3d}/{EPISODES}  Avg100={avg:5.0f} {bar:<20}  Best={best_score}  e={agent.epsilon:.3f}")

    # 最終モデルも保存
    agent.save("pacman_model.pt") if best_score == 0 else None
    print(f"  訓練完了 ({time.time()-t0:.1f}秒)  最高スコア: {best_score}")

    # 学習曲線グラフ保存
    plot_learning_curve(scores, save_path="learning_curve.png")
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
            env2.set_qvalues(agent.get_qvalues(obs))
            obs, _, terminated, truncated, info = env2.step(action)
            if terminated or truncated:
                print(f"  スコア: {info['score']}")
                pygame.time.wait(1000)
                break

    env2.close()
    print("\nデモ完了!")


if __name__ == "__main__":
    main()
