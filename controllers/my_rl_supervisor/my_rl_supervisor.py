import sys
import os

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from ai_supervisor import AISupervisor


def main():
    supervisor = AISupervisor()
    supervisor.run()


if __name__ == "__main__":
    main()


