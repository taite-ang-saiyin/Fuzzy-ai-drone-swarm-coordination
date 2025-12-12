"""Runner to start the realtime visualization connected to Webots telemetry.

Usage:
  python run_realtime_visualization.py

Make sure Webots controllers are started (they send UDP telemetry to base_port + num_drones).
"""
from realtime_simulation import RealtimeDroneSimulation
from live_drone_visualization import LiveDroneVisualization


def main():
    # Create realtime simulation (it starts the telemetry monitor internally)
    sim = RealtimeDroneSimulation()
    viz = LiveDroneVisualization(sim)
    try:
        viz.run(duration=9999.0)
    except KeyboardInterrupt:
        print("Stopping visualization")
    finally:
        sim.stop()


if __name__ == '__main__':
    main()
