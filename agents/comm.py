from __future__ import annotations
import json
import socket
import threading
import time
from typing import Dict, Tuple, List
import copy
import agents.config as C
import numpy as np

def trimf(x: float, abc: list[float]) -> float:
    """Triangular membership function."""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


# =========================
# WorldState: thread-safe storage for drone states
# =========================
class WorldState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}

    def update(self, drone_id: str, state: dict):
        with self._lock:
            self._data[drone_id] = state

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            # Deepcopy ensures nested mutable objects are safe
            return copy.deepcopy(self._data)


# =========================
# UdpPeer: handles sending & receiving drone states
# =========================
class UdpPeer:
    def __init__(
        self,
        self_id: str,
        listen_port: int,
        peer_ports: List[int],
        host: str = C.COMM_HOST
    ):
        self.self_id = self_id
        self.listen_addr: Tuple[str, int] = (host, listen_port)
        self.peer_addrs: List[Tuple[str, int]] = [(host, p) for p in peer_ports]

        # World state for neighbors
        self.world = WorldState()

        # UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.listen_addr)
        timeout = getattr(C, "COMM_TIMEOUT", 0.1)
        self._sock.settimeout(timeout)

        # Thread
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._running = False

        # Buffer size
        self._recv_buf = getattr(C, "COMM_RECV_BUF", 4096)

    def compute_urgency(self, acc_mag: float) -> float:
        """Fuzzy urgency to adjust broadcast interval."""
        # Urgency: low (0-0.5), med (0.3-1.5), high (1.2+)
        urg_high = trimf(acc_mag, [1.2, 2.0, C.MAX_FORCE * 2])
        # High urgency -> shorter interval (multiply by 0.3)
        interval_mult = 1.0 - urg_high * 0.7
        adjusted_interval = (1.0 / C.COMM_HZ) * interval_mult
        return max(0.1, adjusted_interval)  # Min 0.1s to avoid spam

    # -----------------
    # Start/Stop
    # -----------------
    def start(self):
        self._running = True
        self._rx_thread.start()
        print(f"[UdpPeer] {self.self_id} listening on {self.listen_addr}")

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception as e:
            print(f"[UdpPeer] Socket close error: {e}")
        self._rx_thread.join(timeout=1.0)
        print(f"[UdpPeer] {self.self_id} stopped")

    # -----------------
    # Receive loop
    # -----------------
    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self._recv_buf)
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    print(f"[UdpPeer] Socket error: {e}")
                break

            try:
                msg = json.loads(data.decode("utf-8"))
                if "id" in msg and msg["id"] != self.self_id:
                    self.world.update(msg["id"], msg)
            except Exception as e:
                # Ignore malformed packets but log
                print(f"[UdpPeer] Malformed message ignored: {e}")
                continue

    # -----------------
    # Send broadcast
    # -----------------
    def broadcast(self, pos, vel, status="OK"):
        # Convert numpy arrays or lists safely
        pos_list = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
        vel_list = vel.tolist() if isinstance(vel, np.ndarray) else list(vel)

        msg = {
            "id": self.self_id,
            "pos": pos_list,
            "vel": vel_list,
            "status": status,
            "t": time.time(),
        }
        payload = json.dumps(msg).encode("utf-8")
        for addr in self.peer_addrs:
            try:
                self._sock.sendto(payload, addr)
            except Exception as e:
                # UDP is best-effort
                print(f"[UdpPeer] Failed to send to {addr}: {e}")
                continue

    # -----------------
    # Snapshot of all drone states
    # -----------------
    def snapshot(self) -> Dict[str, dict]:
        return self.world.snapshot()
