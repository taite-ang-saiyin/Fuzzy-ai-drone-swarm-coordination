"""
Agents package for the drone swarm simulation.

This package contains:
- comm.py   : UDP communication utilities (peer-to-peer networking)
- boids.py  : Flocking (cohesion, separation, alignment, goal seeking)
- config.py : Global configuration values for drones, comms, and physics
"""

# Explicit exports
from . import comm
from . import boids
from . import config

# So you can do:
#   from agents import comm, boids, config
# or:
#   from agents.comm import UdpPeer

__all__ = ["comm", "boids", "config"]
