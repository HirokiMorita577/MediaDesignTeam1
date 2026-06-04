"""
Pac-Man Custom Gymnasium Environment
カーナビ風経路表示付きパックマン強化学習環境
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import heapq


# マップ定義 (0=通路, 1=壁, 2=エサ, 3=パワーエサ)
DEFAULT_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1],
    [1,3,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,3,1],
    [1,2,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,2,1,1,1,1,1,2,1,2,1,1,2,1],
    [1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,1],
    [1,1,1,1,2,1,1,1,0,1,0,1,1,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,0,0,0,0,0,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,1,1,0,1,1,0,1,2,1,1,1,1],
    [0,0,0,0,2,0,0,1,0,0,0,1,0,0,2,0,0,0,0],
    [1,1,1,1,2,1,0,1,1,1,1,1,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,0,0,0,0,0,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,1,1,1,1,1,0,1,2,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,2,1],
    [1,3,2,1,2,2,2,2,2,0,2,2,2,2,2,1,2,3,1],
    [1,1,2,1,2,1,2,1,1,1,1,1,2,1,2,1,2,1,1],
    [1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,1],
    [1,2,1,1,1,1,1,1,2,1,2,1,1,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

GHOST_COLORS = [(255,0,0), (255,182,255), (0,255,255), (255,165,0)]

CELL_EMPTY = 0
CELL_WALL = 1
CELL_DOT = 2
CELL_POWER = 3

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]


class Ghost:
    def __init__(self, row, col, color):
        self.start = (row, col)
        self.pos = [row, col]
        self.color = color
        self.scared = False
        self.scared_timer = 0
        self.last_pos = None  # 逆戻り防止用

    def reset(self):
        self.pos = list(self.start)
        self.scared = False
        self.scared_timer = 0
        self.last_pos = None

    def _valid_moves(self, grid):
        rows, cols = len(grid), len(grid[0])
        r, c = self.pos
        moves = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL:
                moves.append((nr, nc))
        return moves

    def move(self, grid, pacman_pos, rng):
        valid = self._valid_moves(grid)
        if not valid:
            return

        # 逆戻り禁止（分岐がある場合のみ）
        no_reverse = [p for p in valid if list(p) != self.last_pos]
        candidates = no_reverse if no_reverse else valid

        if self.scared:
            # 逃げる: BFSでパックマンから最も遠いマスへ
            best, best_dist = candidates[0], -1
            for p in candidates:
                dist = abs(p[0] - pacman_pos[0]) + abs(p[1] - pacman_pos[1])
                if dist > best_dist:
                    best_dist = dist
                    best = p
            # 同距離はランダム選択
            farthest = [p for p in candidates if abs(p[0]-pacman_pos[0])+abs(p[1]-pacman_pos[1]) == best_dist]
            chosen = rng.choice(farthest)
        else:
            # 追いかける: BFS最短経路でパックマンへ向かう
            path = _bfs_next(grid, tuple(self.pos), pacman_pos)
            if path and len(path) > 1:
                next_step = path[1]
                # BFS結果がcandidatesにあればそちら優先
                if next_step in candidates:
                    chosen = next_step
                else:
                    # 通常グリーディ（90%）or ランダム（10%）
                    if rng.random() < 0.9:
                        candidates.sort(key=lambda p: abs(p[0]-pacman_pos[0])+abs(p[1]-pacman_pos[1]))
                        chosen = candidates[0]
                    else:
                        chosen = rng.choice(candidates)
            else:
                chosen = rng.choice(candidates)

        self.last_pos = self.pos[:]
        self.pos = list(chosen)

        if self.scared_timer > 0:
            self.scared_timer -= 1
            if self.scared_timer == 0:
                self.scared = False


def _bfs_next(grid, start, goal):
    """BFSで start→goal の最短経路を返す"""
    rows, cols = len(grid), len(grid[0])
    goal = tuple(goal)
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        r, c = pos
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path+[(nr, nc)]))
    return []


def astar(grid, start, goal):
    """A*アルゴリズムで経路探索"""
    rows, cols = len(grid), len(grid[0])
    if grid[goal[0]][goal[1]] == CELL_WALL:
        return []

    def h(p):
        return abs(p[0]-goal[0]) + abs(p[1]-goal[1])

    open_heap = [(h(start), 0, start, [start])]
    visited = set()

    while open_heap:
        f, g, pos, path = heapq.heappop(open_heap)
        if pos in visited:
            continue
        visited.add(pos)
        if pos == goal:
            return path
        r, c = pos
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                new_g = g + 1
                heapq.heappush(open_heap, (new_g + h((nr,nc)), new_g, (nr,nc), path+[(nr,nc)]))
    return []


def find_nearest_dot(grid, pos):
    """最近傍のエサをBFSで探す"""
    rows, cols = len(grid), len(grid[0])
    queue = deque([(pos, [pos])])
    visited = {pos}
    while queue:
        (r, c), path = queue.popleft()
        if grid[r][c] in (CELL_DOT, CELL_POWER):
            return (r, c), path
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path+[(nr, nc)]))
    return None, []


class PacManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.base_map = [row[:] for row in DEFAULT_MAP]
        self.rows = len(self.base_map)
        self.cols = len(self.base_map[0])
        self.render_mode = render_mode
        self.cell_size = 28
        self.screen = None
        self.clock = None
        self.rng = np.random.default_rng()

        # 観測空間: グリッド状態(4チャネル) + パックマン位置 + ゴースト位置
        obs_size = self.rows * self.cols * 4 + 2 + 4*2
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # 経路表示用
        self.planned_path = []
        self.target_pos = None

    def _init_game(self):
        self.grid = [row[:] for row in self.base_map]
        self.pacman = [16, 9]  # スタート位置
        self.total_dots = sum(1 for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] in (CELL_DOT, CELL_POWER))
        self.dots_eaten = 0
        self.score = 0
        self.power_mode = False
        self.power_timer = 0
        # ゴースト初期位置: ゴーストハウス内の通路セルに配置
        self.ghosts = [
            Ghost(10, 9,  GHOST_COLORS[0]),  # 中央
            Ghost(10, 8,  GHOST_COLORS[1]),  # 左
            Ghost(10, 10, GHOST_COLORS[2]),  # 右
            Ghost(12, 9,  GHOST_COLORS[3]),  # 下段
        ]
        self.steps = 0
        self.planned_path = []
        self.target_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_game()
        self._update_planned_path()
        return self._get_obs(), {}

    def _get_obs(self):
        # 4チャネル: 壁/エサ/パワーエサ/ゴースト
        ch_wall = np.array([[1 if c == CELL_WALL else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        ch_dot  = np.array([[1 if c == CELL_DOT else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        ch_pow  = np.array([[1 if c == CELL_POWER else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        gh_map  = np.zeros((self.rows, self.cols), dtype=np.float32)
        for g in self.ghosts:
            gh_map[g.pos[0], g.pos[1]] = 1.0
        ch_ghost = gh_map.flatten()

        pacman_norm = np.array([self.pacman[0]/self.rows, self.pacman[1]/self.cols], dtype=np.float32)
        ghost_norm  = np.array([c for g in self.ghosts for c in [g.pos[0]/self.rows, g.pos[1]/self.cols]], dtype=np.float32)
        return np.concatenate([ch_wall, ch_dot, ch_pow, ch_ghost, pacman_norm, ghost_norm])

    def _update_planned_path(self):
        """カーナビ風: 最近傍のエサへのA*経路を計算"""
        pos = tuple(self.pacman)
        target, _ = find_nearest_dot(self.grid, pos)
        if target:
            self.target_pos = target
            self.planned_path = astar(self.grid, pos, target)
        else:
            self.planned_path = []
            self.target_pos = None

    def step(self, action):
        self.steps += 1
        dr, dc = DIRS[action]
        nr, nc = self.pacman[0]+dr, self.pacman[1]+dc

        reward = -0.01  # 時間ペナルティ

        # 壁チェック: 壁にぶつかったら罰
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != CELL_WALL:
            self.pacman = [nr, nc]
        else:
            reward -= 0.5  # 壁衝突ペナルティ

        r, c = self.pacman

        # エサ取得
        if self.grid[r][c] == CELL_DOT:
            self.grid[r][c] = CELL_EMPTY
            self.dots_eaten += 1
            self.score += 10
            reward += 1.0

        elif self.grid[r][c] == CELL_POWER:
            self.grid[r][c] = CELL_EMPTY
            self.dots_eaten += 1
            self.score += 50
            reward += 2.0
            self.power_mode = True
            self.power_timer = 30
            for g in self.ghosts:
                g.scared = True
                g.scared_timer = 30

        # パワーモード更新
        if self.power_timer > 0:
            self.power_timer -= 1
            if self.power_timer == 0:
                self.power_mode = False

        # ゴースト移動
        for g in self.ghosts:
            g.move(self.grid, tuple(self.pacman), self.rng)

        # ゴースト衝突
        terminated = False
        for g in self.ghosts:
            if g.pos == self.pacman:
                if g.scared:
                    g.reset()
                    self.score += 200
                    reward += 5.0
                else:
                    reward -= 10.0
                    terminated = True

        # クリア
        if self.dots_eaten >= self.total_dots:
            reward += 50.0
            terminated = True

        truncated = self.steps >= 1000

        # 経路更新 (5ステップ毎)
        if self.steps % 5 == 0:
            self._update_planned_path()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame
        cs = self.cell_size
        W = self.cols * cs
        H = self.rows * cs + 60  # 情報バー分

        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Pac-Man RL - カーナビ風経路表示")
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((W, H))
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14, bold=True)
            self.font_big = pygame.font.SysFont("Arial", 18, bold=True)

        self.screen.fill((0, 0, 0))

        # ── グリッド描画 ──
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c*cs, r*cs
                cell = self.grid[r][c]
                if cell == CELL_WALL:
                    pygame.draw.rect(self.screen, (30, 80, 200), (x, y, cs, cs))
                    pygame.draw.rect(self.screen, (50, 100, 220), (x+1, y+1, cs-2, cs-2), 1)
                elif cell == CELL_DOT:
                    pygame.draw.circle(self.screen, (255, 220, 150), (x+cs//2, y+cs//2), 3)
                elif cell == CELL_POWER:
                    t = pygame.time.get_ticks() // 300 % 2
                    col = (255, 200, 50) if t == 0 else (255, 100, 0)
                    pygame.draw.circle(self.screen, col, (x+cs//2, y+cs//2), 7)

        # ── カーナビ経路表示 ──
        path = self.planned_path
        if len(path) > 1:
            # 経路を半透明の帯で描画
            surf = pygame.Surface((W, self.rows*cs), pygame.SRCALPHA)
            for i in range(len(path)-1):
                r0,c0 = path[i]
                r1,c1 = path[i+1]
                x0,y0 = c0*cs+cs//2, r0*cs+cs//2
                x1,y1 = c1*cs+cs//2, r1*cs+cs//2
                # 進捗に応じて色変化 (青→緑)
                t = i / max(len(path)-1, 1)
                color = (int(0+100*t), int(180-80*t), int(255-100*t), 160)
                pygame.draw.line(surf, color, (x0,y0), (x1,y1), 6)
            self.screen.blit(surf, (0,0))

            # ウェイポイント矢印
            for i in range(0, len(path)-1, 2):
                r0,c0 = path[i]
                r1,c1 = path[i+1]
                cx,cy = c0*cs+cs//2, r0*cs+cs//2
                dr,dc = r1-r0, c1-c0
                arrow_pts = self._arrow_points(cx, cy, dc*cs//2, dr*cs//2, 5)
                pygame.draw.polygon(self.screen, (0, 255, 180), arrow_pts)

        # 目標地点マーカー
        if self.target_pos:
            tr, tc = self.target_pos
            tx, ty = tc*cs+cs//2, tr*cs+cs//2
            pygame.draw.circle(self.screen, (255, 50, 50), (tx, ty), 8, 3)
            pygame.draw.line(self.screen, (255, 50, 50), (tx, ty-12), (tx, ty-4), 2)

        # ── ゴースト描画 ──
        for g in self.ghosts:
            gr, gc = g.pos
            gx, gy = gc*cs+cs//2, gr*cs+cs//2
            color = (100, 100, 255) if g.scared else g.color
            pygame.draw.circle(self.screen, color, (gx, gy), cs//2-3)
            pygame.draw.rect(self.screen, color, (gc*cs+3, gr*cs+cs//2, cs-6, cs//2-3))
            # 目
            ew = 4 if not g.scared else 3
            ec = (255,255,255) if not g.scared else (255,255,100)
            pygame.draw.circle(self.screen, ec, (gx-5, gy-3), ew)
            pygame.draw.circle(self.screen, ec, (gx+5, gy-3), ew)
            if not g.scared:
                pygame.draw.circle(self.screen, (0,0,100), (gx-4, gy-3), 2)
                pygame.draw.circle(self.screen, (0,0,100), (gx+6, gy-3), 2)

        # ── パックマン描画 ──
        pr, pc = self.pacman
        px, py = pc*cs+cs//2, pr*cs+cs//2
        angle = (pygame.time.get_ticks() // 100 % 4) * 10 + 10
        color_pac = (255, 255, 0) if not self.power_mode else (255, 150, 0)
        pygame.draw.circle(self.screen, color_pac, (px, py), cs//2-2)
        # 口
        mouth_pts = [(px, py)]
        for a in range(angle, 360-angle):
            rad = np.radians(a)
            mouth_pts.append((px + int((cs//2-2)*np.cos(rad)), py + int((cs//2-2)*np.sin(rad))))
        if len(mouth_pts) > 2:
            pygame.draw.polygon(self.screen, (0, 0, 0), mouth_pts)

        # ── 情報バー (カーナビ風) ──
        info_y = self.rows * cs
        pygame.draw.rect(self.screen, (20, 20, 40), (0, info_y, W, 60))
        pygame.draw.line(self.screen, (0, 200, 255), (0, info_y), (W, info_y), 2)

        # スコア
        s = self.font_big.render(f"SCORE: {self.score}", True, (0, 255, 200))
        self.screen.blit(s, (10, info_y+8))

        # 経路情報
        path_len = len(self.planned_path)
        nav_text = f"経路: {path_len}歩  目標: {self.target_pos}" if self.target_pos else "経路探索中..."
        nav_col = (0, 220, 255) if path_len > 0 else (255, 150, 0)
        ns = self.font.render(nav_text, True, nav_col)
        self.screen.blit(ns, (10, info_y+32))

        # パワーモード表示
        if self.power_mode:
            pt = self.font.render(f"⚡POWER MODE! ({self.power_timer})", True, (255, 220, 0))
            self.screen.blit(pt, (W//2-60, info_y+8))

        # エサ残数
        remain = self.total_dots - self.dots_eaten
        ds = self.font.render(f"残エサ: {remain}/{self.total_dots}", True, (200, 200, 200))
        self.screen.blit(ds, (W-130, info_y+8))

        # ステップ
        ss = self.font.render(f"Step: {self.steps}", True, (150, 150, 150))
        self.screen.blit(ss, (W-130, info_y+30))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def _arrow_points(self, x, y, dx, dy, size):
        import math
        angle = math.atan2(dy, dx)
        pts = [
            (x + dx, y + dy),
            (x + dx - size*math.cos(angle-0.4), y + dy - size*math.sin(angle-0.4)),
            (x + dx - size*math.cos(angle+0.4), y + dy - size*math.sin(angle+0.4)),
        ]
        return pts

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
