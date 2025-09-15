# comm.py
from __future__ import annotations
import json
import socket
import threading
import time
from typing import Dict, Tuple
import numpy as np
import final_config as C

def trimf(x: float, abc: list[float]) -> float:
    """Triangular membership function."""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

class WorldState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}

    def update(self, drone_id: str, state: dict):
        with self._lock:
            self._data[drone_id] = state

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            # Return a shallow copy to avoid locking during compute
            return dict(self._data)

class UdpPeer:
    def __init__(self, self_id: str, listen_port: int, peer_ports: list[int], host: str = C.COMM_HOST):
        self.self_id = self_id
        self.listen_addr: Tuple[str, int] = (host, listen_port)
        self.peer_addrs: list[Tuple[str, int]] = [(host, p) for p in peer_ports]
        self.world = WorldState()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.listen_addr)
        self._sock.settimeout(C.COMM_TIMEOUT)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._running = False

    def compute_urgency(self, acc_mag: float) -> float:
        """Fuzzy urgency to adjust broadcast interval."""
        # Urgency: low (0-0.5), med (0.3-1.5), high (1.2+)
        urg_high = trimf(acc_mag, [1.2, 2.0, C.MAX_FORCE * 2])
        # High urgency -> shorter interval (multiply by 0.3)
        interval_mult = 1.0 - urg_high * 0.7
        adjusted_interval = (1.0 / C.COMM_HZ) * interval_mult
        return max(0.1, adjusted_interval)  # Min 0.1s to avoid spam

    def start(self):
        self._running = True
        self._rx_thread.start()

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(C.COMM_RECV_BUF)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                if "id" in msg and msg["id"] != self.self_id:
                    self.world.update(msg["id"], msg)
            except Exception:
                # Ignore malformed packets
                continue

    def broadcast(self, pos, vel, status="OK"):
        msg = {
            "id": self.self_id,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "vel": [float(vel[0]), float(vel[1]), float(vel[2])],
            "status": status,
            "t": time.time(),
        }
        payload = json.dumps(msg).encode("utf-8")
        for addr in self.peer_addrs:
            try:
                self._sock.sendto(payload, addr)
            except Exception:
                # Best-effort; UDP may drop
                continue

    def snapshot(self):
        return self.world.snapshot()