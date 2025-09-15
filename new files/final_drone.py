# drone.py
from __future__ import annotations
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

import final_config as C
import final_boids
from final_comm import UdpPeer

@dataclass
class DroneConfig:
    drone_id: str
    listen_port: int
    peer_ports: List[int]
    goal: Optional[np.ndarray] = None

class DroneAgent:
    def __init__(self, cfg: DroneConfig):
        self.id = cfg.drone_id
        self.comm = UdpPeer(self.id, cfg.listen_port, cfg.peer_ports)
        self.goal = cfg.goal

        # Initialize state in a ring
        idx = int(self.id.split("_")[-1])
        theta = (2 * math.pi * idx) / max(1, (len(cfg.peer_ports) + 1))
        self.pos = np.array([
            C.INIT_RING_RADIUS * math.cos(theta),
            C.INIT_RING_RADIUS * math.sin(theta),
            C.INIT_ALTITUDE
        ], dtype=float)
        # Small tangential initial velocity + common forward direction for pack movement
        self.vel = np.array([
            -math.sin(theta), math.cos(theta), 0.0
        ], dtype=float) * 0.5 + np.array([2.0, 0.0, 0.0])  # Common x-direction
        self.vel = final_boids._limit(self.vel, C.MAX_SPEED)  # Clamp

        # Logging
        os.makedirs(C.LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(C.LOG_DIR, f"{self.id}.csv")
        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "z", "vx", "vy", "vz"])

    def run(self, duration_s: float):
        self.comm.start()
        t0 = time.time()
        last_tx = t0
        last_heartbeat = t0
        try:
            while True:
                now = time.time()
                if now - t0 >= duration_s:
                    break

                world = self.comm.snapshot()

                # Compute steering using boids
                acc = final_boids.step(self.pos, self.vel, world, self.goal)
                self.pos, self.vel = final_boids.integrate(self.pos, self.vel, acc, C.TIME_STEP)

                # Broadcast with fuzzy urgency-adjusted interval
                acc_mag = np.linalg.norm(acc)
                urgency_interval = self.comm.compute_urgency(acc_mag)
                if now - last_tx >= urgency_interval:
                    self.comm.broadcast(self.pos, self.vel)
                    last_tx = now

                # Log
                with open(self.log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([now, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2]])

                # Heartbeat
                if now - last_heartbeat >= 1.0:
                    n = len(world)
                    print(f"[{self.id}] t={now - t0:5.1f}s | neighbors={n} | pos=({self.pos[0]:.2f},{self.pos[1]:.2f},{self.pos[2]:.2f})")
                    last_heartbeat = now

                time.sleep(max(0.0, C.TIME_STEP - (time.time() - now)))
        finally:
            self.comm.stop()