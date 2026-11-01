from __future__ import annotations
from collections import deque
import numpy as np


class StallDetector:
    def __init__(self, cfg):
        self.window = float(cfg["termination"]["stall_time"])
        self.disp = float(cfg["termination"]["stall_displacement"])
        self.progress = float(cfg["termination"]["stall_goal_progress"])
        self.speed = float(cfg["termination"]["stall_speed"])
        self.data = deque()

    def update(self, t, p, goal, speed):
        gd = float(np.linalg.norm(np.asarray(goal) - np.asarray(p)))
        self.data.append((float(t), np.asarray(p, float).copy(), gd, float(speed)))
        while self.data and t - self.data[0][0] > self.window:
            self.data.popleft()
        if not self.data or t - self.data[0][0] < 0.95 * self.window:
            return False
        displacement = float(np.linalg.norm(self.data[-1][1] - self.data[0][1]))
        goal_progress = float(self.data[0][2] - self.data[-1][2])
        mean_speed = float(np.mean([x[3] for x in self.data]))
        return (
            displacement < self.disp
            and goal_progress < self.progress
            and mean_speed < self.speed
        )
