"""
カーナビ風経路表示でAIプレイを観戦
使い方:
  python play.py                             # 訓練済みモデルで自動プレイ
  python play.py --record                    # 録画（0.4倍速で見やすく）
  python play.py --record --speed 0.3        # さらにゆっくり録画
  python play.py --record --out game.mp4     # 出力ファイル名を指定
  python play.py --human                     # 手動プレイ (矢印キー)
  python play.py --episodes 5                # 5ゲーム連続プレイ

録画スピードについて:
  --speed 1.0  通常速度
  --speed 0.5  半速（録画デフォルト）
  --speed 0.3  ゆっくり（解説向き）
  動画は常に --video-fps で滑らかに出力（フレーム複製方式）
"""
import argparse
import os
import datetime
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


class ScreenRecorder:
    """
    pygame 画面を MP4 として録画するクラス

    【フレーム複製方式】
      ゲームのステップレートを speed 倍に落としつつ、
      動画は video_fps で滑らかに出力する。
      1ゲームステップあたり repeat_frames フレームを記録することで
      動画の再生速度 = ゲームの見た目速度 を一致させる。

      例: game_fps=5 (speed=0.5), video_fps=15 → 1ステップ=3フレーム複製
    """

    def __init__(self, path, video_fps, game_fps, width, height):
        import cv2
        self.path = path
        self.video_fps = video_fps
        self.game_fps  = game_fps
        # 1ゲームステップで何フレーム書き込むか
        self.repeat = max(1, round(video_fps / game_fps))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, video_fps, (width, height))
        self.frame_count = 0
        print(f"[録画開始] -> {path}  ({width}x{height})")
        print(f"  動画fps={video_fps}  ゲームfps={game_fps:.1f}  "
              f"1ステップ={self.repeat}フレーム複製")

    def capture(self, surface, repeat=None):
        """pygame Surface を録画（repeat=None で self.repeat 回複製）"""
        import cv2
        n = repeat if repeat is not None else self.repeat
        raw = pygame.surfarray.array3d(surface)
        frame = np.transpose(raw, (1, 0, 2))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for _ in range(n):
            self.writer.write(frame_bgr)
            self.frame_count += 1

    def stop(self):
        self.writer.release()
        duration = self.frame_count / self.video_fps
        print(f"[録画終了] {self.frame_count}フレーム  "
              f"動画長さ約{duration:.1f}秒 -> {self.path}")


def play(args):
    env = PacManEnv(render_mode="human", ghost_mode=args.ghost_mode)
    obs_size  = env.observation_space.shape[0]
    n_actions = env.action_space.n
    base_fps  = env.metadata["render_fps"]   # 環境の標準fps

    # 録画時はスピードを落とす（デフォルト 0.5 倍速）
    speed = args.speed
    if args.record and speed == 1.0 and not args.no_slowdown:
        speed = 0.5   # 録画時はデフォルトで半速

    # ゲームステップレート（実際にwaitする fps）
    game_fps = base_fps * speed
    # pygame クロック
    clock = pygame.time.Clock()

    agent = None
    if not args.human:
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions,
                         epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0)
        model_path = args.model if args.model != "none" else "pacman_model.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"モデル読込: {model_path}")
        else:
            print(f"[警告] モデルファイル '{model_path}' が見つかりません。ランダムで動作します。")
            agent.epsilon = 1.0

    # 録画セットアップ
    recorder = None
    if args.record:
        obs, _ = env.reset()
        env.render()
        pygame.display.flip()
        w = env.screen.get_width()
        h = env.screen.get_height()

        out_path = args.out
        if out_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"replay_{ts}.mp4"

        video_fps = args.video_fps
        recorder = ScreenRecorder(out_path, video_fps, game_fps, w, h)
        first_reset_done = True

        print(f"  速度: {speed:.1f}x  ({game_fps:.1f} steps/sec)")
    else:
        first_reset_done = False
        if speed != 1.0:
            print(f"  速度: {speed:.1f}x")

    total_scores = []

    for ep in range(args.episodes):
        if first_reset_done:
            first_reset_done = False
        else:
            obs, _ = env.reset()
            env.render()

        total_reward = 0
        action = 3

        print(f"\n=== Episode {ep+1}/{args.episodes} ===")
        if args.record:
            print(f"  [REC] 録画中... ({speed:.1f}x速)")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if recorder:
                        recorder.stop()
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if recorder:
                            recorder.stop()
                        env.close()
                        return
                    if args.human and event.key in ACTION_MAP:
                        action = ACTION_MAP[event.key]

            if not args.human and agent:
                action = agent.select_action(obs, training=False)
                env.set_qvalues(agent.get_qvalues(obs))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # フレームキャプチャ（1ステップ分を repeat 回複製して動画速度を調整）
            if recorder and env.screen:
                recorder.capture(env.screen)

            # ゲーム速度制御（録画あり・なし共通）
            clock.tick(game_fps)

            if terminated or truncated:
                running = False

        score = info["score"]
        total_scores.append(score)
        print(f"スコア: {score}  累積報酬: {total_reward:.1f}")

        if env.screen:
            # ゲームオーバー後 2秒間静止フレーム録画
            still_frames = int(args.video_fps * 2) if args.record else 0
            for _ in range(still_frames):
                pygame.event.pump()
                if recorder:
                    recorder.capture(env.screen, repeat=1)
            pygame.time.wait(800)

    if recorder:
        recorder.stop()

    if total_scores:
        print(f"\n{'='*40}")
        print(f"全{args.episodes}ゲーム完了")
        print(f"平均スコア: {np.mean(total_scores):.1f}")
        print(f"最高スコア: {max(total_scores)}")
        print(f"最低スコア: {min(total_scores)}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,   default=3)
    parser.add_argument("--human",       action="store_true", help="手動プレイモード")
    parser.add_argument("--model",       type=str,   default="pacman_model.pt")
    parser.add_argument("--record",      action="store_true", help="MP4録画を有効にする")
    parser.add_argument("--out",         type=str,   default=None,
                        help="出力ファイル名 (デフォルト: replay_日時.mp4)")
    parser.add_argument("--speed",       type=float, default=1.0,
                        help="ゲーム速度倍率 (録画時デフォルト0.5, 例: 0.3でゆっくり)")
    parser.add_argument("--video-fps",   type=int,   default=15,
                        help="録画動画のFPS (デフォルト15, 高いほど滑らか)")
    parser.add_argument("--no-slowdown", action="store_true",
                        help="録画時も速度を落とさない")
    parser.add_argument("--ghost-mode", type=str, default="classic",
                        choices=["classic", "bfs"],
                        help="ゴーストAI: classic=本家4体行動(デフォルト), bfs=全員直接追跡")
    args = parser.parse_args()
    play(args)
