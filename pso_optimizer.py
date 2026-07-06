"""
粒子群最適化 (Particle Swarm Optimization: PSO) 実装
================================================================

【PSO理論】

PSOはKennedy & Eberhartが1995年に提案したメタヒューリスティック最適化手法です。
鳥の群れや魚の群泳といった集団知性(Swarm Intelligence)を模倣します。

■ 基本概念
  - 粒子(Particle): 解空間上の1点(=ハイパーパラメータの組み合わせ)
  - 群れ(Swarm)   : 複数の粒子の集合
  - 適応度(Fitness): その粒子が表す解の良さ(=パックマンの平均スコア)
  - pbest         : 各粒子がこれまでに見つけた最良の位置
  - gbest         : 全粒子を通じて見つかった最良の位置

■ 更新式
  各粒子iの速度vと位置xを以下のルールで更新します:

    v[i] = w * v[i]
          + c1 * r1 * (pbest[i] - x[i])   ← 認知項(自分の経験)
          + c2 * r2 * (gbest   - x[i])    ← 社会項(群れの知識)

    x[i] = x[i] + v[i]

  w  : 慣性重み    (inertia weight)     → 過去の速度を維持する度合い
  c1 : 認知係数    (cognitive coeff.)   → 自己の最良位置へ引き寄せる強さ
  c2 : 社会係数    (social coeff.)      → 群れの最良位置へ引き寄せる強さ
  r1, r2: [0,1]の乱数                   → 確率的な探索を実現

■ 慣性重み w の役割
  w が大きい (>0.9): 広い探索 (exploration) を優先
  w が小さい (<0.4): 局所的な精密化 (exploitation) を優先
  → 線形減衰: w = w_max - (w_max - w_min) * t/T で両立

■ 探索空間
  本実装では以下のハイパーパラメータを最適化します:
    - learning_rate : ニューラルネットの更新ステップ幅
    - epsilon_end   : epsilon-greedyの最終探索率
    - epsilon_decay : epsilonの減衰率（1ステップあたり）
    - gamma         : 割引率（将来の報酬をどれだけ重視するか）

■ 対数スケール変換
  学習率などは桁が変わる量のため、log空間で探索します:
    実際の値 = 10^(粒子の位置)
  例: 粒子位置=-3 → 学習率=0.001

================================================================
"""

import numpy as np


class Particle:
    """1つの粒子: ハイパーパラメータ空間上の点"""

    def __init__(self, bounds, rng):
        """
        bounds: [(min1,max1), (min2,max2), ...] 各次元の探索範囲
        """
        self.dim = len(bounds)
        self.bounds = np.array(bounds, dtype=np.float64)

        # 位置をランダム初期化
        self.pos = rng.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # 速度を探索範囲の10%でランダム初期化
        span = self.bounds[:, 1] - self.bounds[:, 0]
        self.vel = rng.uniform(-span * 0.1, span * 0.1)

        self.pbest_pos = self.pos.copy()
        self.pbest_score = -np.inf

    def update_velocity(self, gbest_pos, w, c1, c2, rng):
        """速度更新式: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)"""
        r1 = rng.random(self.dim)
        r2 = rng.random(self.dim)
        cognitive = c1 * r1 * (self.pbest_pos - self.pos)
        social    = c2 * r2 * (gbest_pos      - self.pos)
        self.vel  = w * self.vel + cognitive + social

        # 速度クリッピング: 探索範囲の20%以内に制限
        span = self.bounds[:, 1] - self.bounds[:, 0]
        self.vel = np.clip(self.vel, -span * 0.2, span * 0.2)

    def update_position(self):
        """位置更新式: x = x + v (境界でクリップ)"""
        self.pos = np.clip(self.pos + self.vel,
                           self.bounds[:, 0], self.bounds[:, 1])

    def update_pbest(self, score):
        """個人最良解の更新"""
        if score > self.pbest_score:
            self.pbest_score = score
            self.pbest_pos   = self.pos.copy()
            return True
        return False


class PSO:
    """
    粒子群最適化エンジン

    パラメータ:
        n_particles : 粒子数 (多いほど広く探索、計算コスト増)
        n_iterations: 世代数 (多いほど収束精度向上)
        w_start     : 初期慣性重み (大きい=広い探索)
        w_end       : 最終慣性重み (小さい=精密化)
        c1          : 認知係数
        c2          : 社会係数
    """

    def __init__(self, n_particles=10, n_iterations=10,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
                 seed=42):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w_start      = w_start
        self.w_end        = w_end
        self.c1           = c1
        self.c2           = c2
        self.rng          = np.random.default_rng(seed)

        self.gbest_pos    = None
        self.gbest_score  = -np.inf
        self.history      = []   # (iteration, gbest_score, gbest_params)

    def optimize(self, fitness_fn, bounds, param_names=None, verbose=True):
        """
        fitness_fn  : 粒子の位置を受け取りスコアを返す関数
        bounds      : 各次元の探索範囲リスト
        param_names : パラメータ名リスト（表示用）
        戻り値      : (最良パラメータ dict, 履歴 list)
        """
        if param_names is None:
            param_names = [f"x{i}" for i in range(len(bounds))]

        # 粒子群の初期化
        swarm = [Particle(bounds, self.rng) for _ in range(self.n_particles)]

        if verbose:
            print("\n" + "="*60)
            print("  PSO ハイパーパラメータ最適化 開始")
            print(f"  粒子数: {self.n_particles}  世代数: {self.n_iterations}")
            print(f"  慣性重み: {self.w_start} → {self.w_end}  c1={self.c1}  c2={self.c2}")
            print("="*60)
            print(f"\n  探索空間:")
            for name, (lo, hi) in zip(param_names, bounds):
                print(f"    {name:<20}: [{lo:.4f}, {hi:.4f}]")
            print()

        # 世代ループ
        for t in range(self.n_iterations):
            # 線形減衰慣性重み
            w = self.w_start - (self.w_start - self.w_end) * t / max(self.n_iterations - 1, 1)

            if verbose:
                print(f"[世代 {t+1:2d}/{self.n_iterations}]  w={w:.3f}")

            # 各粒子の評価
            for i, p in enumerate(swarm):
                score = fitness_fn(p.pos)

                # 個人最良更新
                updated = p.update_pbest(score)

                # 群れ最良更新
                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos   = p.pos.copy()

                if verbose:
                    mark = " *gbest*" if score == self.gbest_score else (
                           " +pbest+" if updated else "")
                    params_str = "  ".join(
                        f"{n}={v:.4f}" for n, v in zip(param_names, p.pos))
                    print(f"  粒子{i+1:2d}: score={score:6.1f}  [{params_str}]{mark}")

            # 速度・位置の更新
            for p in swarm:
                p.update_velocity(self.gbest_pos, w, self.c1, self.c2, self.rng)
                p.update_position()

            self.history.append({
                "iteration"    : t + 1,
                "gbest_score"  : self.gbest_score,
                "gbest_pos"    : self.gbest_pos.copy(),
                "pbest_scores" : [p.pbest_score for p in swarm],  # 全粒子のpbest
                "cur_scores"   : [p.pbest_score for p in swarm],  # 現世代スコア
            })

            if verbose:
                params_str = "  ".join(
                    f"{n}={v:.4f}" for n, v in zip(param_names, self.gbest_pos))
                print(f"  >> 現在の最良スコア: {self.gbest_score:.1f}  [{params_str}]\n")

        # 結果をdict形式で返す
        best_params = {name: val for name, val in zip(param_names, self.gbest_pos)}

        if verbose:
            print("="*60)
            print("  PSO 最適化完了")
            print(f"  最良スコア: {self.gbest_score:.1f}")
            print("  最良パラメータ:")
            for name, val in best_params.items():
                print(f"    {name:<20}: {val:.6f}")
            print("="*60 + "\n")

        return best_params, self.history
