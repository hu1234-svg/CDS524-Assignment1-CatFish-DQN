import argparse
import math
import os
import random

import numpy as np
import pygame

import torch
import torch.nn as nn



W, H = 720, 520
FPS = 60

CAT_R = 14
CAT_SPEED = 4

FISH_R = 10
TRAP_R = 12
N_TRAPS = 5
MAX_STEPS = 900  

MODEL_PATH = "dqn_catfish.pt"

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def rand_pos(m=60):
    return random.randint(m, W - m), random.randint(m, H - m)


def action_to_vec(a):
    
    if a == 0:
        return (0, -CAT_SPEED)
    if a == 1:
        return (0, CAT_SPEED)
    if a == 2:
        return (-CAT_SPEED, 0)
    if a == 3:
        return (CAT_SPEED, 0)
    return (0, 0)


def get_human_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return 1
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 2
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 3
    return 4


# =========================
# Q Network (MUST match Colab)
# =========================
class QNet(nn.Module):
    def __init__(self, state_dim=11, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def nearest_trap(cx, cy, traps):
    best_d = 1e9
    best = None
    for tx, ty in traps:
        d = dist(cx, cy, tx, ty)
        if d < best_d:
            best_d = d
            best = (tx, ty)
    return best[0], best[1], best_d


def build_state(cx, cy, cvx, cvy, fx, fy, traps, step):
    """
    EXACT match with Colab env._get_state()

    State vector (11 dims):
    [cat_x, cat_y, cat_vx, cat_vy,
     fish_dx, fish_dy, fish_dist,
     nearest_trap_dx, nearest_trap_dy, nearest_trap_dist,
     step_frac]
    """
    cat_x = cx / W
    cat_y = cy / H

    # Colab: cat_vx = self.cvx / CAT_SPEED
    cat_vx = (cvx / CAT_SPEED) if CAT_SPEED > 0 else 0.0
    cat_vy = (cvy / CAT_SPEED) if CAT_SPEED > 0 else 0.0

    fish_dx = (fx - cx) / W
    fish_dy = (fy - cy) / H
    fish_d = dist(cx, cy, fx, fy) / math.hypot(W, H)

    ntx, nty, td = nearest_trap(cx, cy, traps)
    trap_dx = (ntx - cx) / W
    trap_dy = (nty - cy) / H
    trap_d = td / math.hypot(W, H)

    step_frac = step / MAX_STEPS

    return np.array(
        [
            cat_x,
            cat_y,
            cat_vx,
            cat_vy,
            fish_dx,
            fish_dy,
            fish_d,
            trap_dx,
            trap_dy,
            trap_d,
            step_frac,
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def dqn_greedy_action(model, state_np):
    s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)  # (1,11)
    q = model(s)  # (1,5)
    return int(q.argmax(dim=1).item())


def main(mode="human"):
    pygame.init()
    pygame.display.set_caption("Cat Eats Fish (Local UI + DQN)")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big = pygame.font.SysFont("consolas", 28)

    # ---- load DQN model (play mode only) ----
    model = None
    if mode == "play":
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: {MODEL_PATH} not found in this folder.")
            print("Fix: put dqn_catfish.pt in the same folder as main.py, then rerun.")
            return
        model = QNet(state_dim=11, n_actions=5)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

    # ---- game state ----
    cx, cy = W // 2, H // 2
    cvx, cvy = 0.0, 0.0  # IMPORTANT: match Colab state includes velocity

    fx, fy = rand_pos()
    traps = [rand_pos() for _ in range(N_TRAPS)]

    score = 0
    step = 0
    last_action = "STAY"
    last_reward = 0.0

    # ---- effects ----
    spark_timer = 0
    float_timer = 0
    float_pos = (0, 0)

    running = True
    done = False

    while running:
        clock.tick(FPS)

        # ---- events ----
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                # restart
                cx, cy = W // 2, H // 2
                cvx, cvy = 0.0, 0.0
                fx, fy = rand_pos()
                traps = [rand_pos() for _ in range(N_TRAPS)]
                score = 0
                step = 0
                last_action = "STAY"
                last_reward = 0.0
                done = False

        if not done:
            step += 1
            last_reward = 0.0

            # ---- choose action ----
            if mode == "human":
                a = get_human_action()
            else:
                # build state (exact match with training)
                s = build_state(cx, cy, cvx, cvy, fx, fy, traps, step)
                a = dqn_greedy_action(model, s)

            last_action = ACTIONS[a]

            # ---- move cat (and update velocity like Colab) ----
            dx, dy = action_to_vec(a)
            cvx, cvy = float(dx), float(dy)  # Colab: self.cvx, self.cvy = dx*CAT_SPEED, dy*CAT_SPEED

            cx = max(CAT_R, min(W - CAT_R, cx + dx))
            cy = max(CAT_R, min(H - CAT_R, cy + dy))

            # ---- eat fish ----
            if dist(cx, cy, fx, fy) <= CAT_R + FISH_R:
                score += 1
                last_reward = 1.0
                fx, fy = rand_pos()

                spark_timer = 10
                float_timer = 30
                float_pos = (cx, cy)

            # ---- hit trap -> done ----
            for tx, ty in traps:
                if dist(cx, cy, tx, ty) <= CAT_R + TRAP_R:
                    last_reward = -10.0
                    done = True
                    break

            if step >= MAX_STEPS:
                done = True

        # ---- draw ----
        screen.fill((16, 16, 18))
        pygame.draw.rect(screen, (80, 80, 90), pygame.Rect(10, 10, W - 20, H - 20), 2)

        # fish
        pygame.draw.circle(screen, (80, 170, 255), (fx, fy), FISH_R)
        # traps
        for tx, ty in traps:
            pygame.draw.circle(screen, (220, 70, 70), (tx, ty), TRAP_R)
        # cat
        pygame.draw.circle(screen, (120, 240, 160), (int(cx), int(cy)), CAT_R)

        # eat effect
        if spark_timer > 0:
            spark_timer -= 1
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, int(120 * (spark_timer / 10))))
            screen.blit(overlay, (0, 0))

        if float_timer > 0:
            float_timer -= 1
            x, y = float_pos
            dy = int((30 - float_timer) * 0.8)
            txt = big.render("Yummy! +1", True, (255, 235, 120))
            screen.blit(txt, (x - txt.get_width() // 2, y - 60 - dy))

        # HUD
        hud_lines = [
            f"Mode: {mode}" + (" (DQN greedy)" if mode == "play" else ""),
            f"Step: {step}/{MAX_STEPS}",
            f"Score: {score}",
            f"Last Action: {last_action}",
            f"Last Reward: {last_reward:+.1f}",
            "ESC quit | R restart",
        ]
        y = 16
        for t in hud_lines:
            screen.blit(font.render(t, True, (230, 230, 230)), (16, y))
            y += 20

        if done:
            go = big.render("DONE (press R to restart)", True, (255, 140, 140))
            screen.blit(go, (W // 2 - go.get_width() // 2, H // 2 - go.get_height() // 2))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "play"], default="human")
    args = parser.parse_args()
    main(mode=args.mode)
