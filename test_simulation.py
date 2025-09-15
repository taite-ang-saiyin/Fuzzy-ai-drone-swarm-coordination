#!/usr/bin/env python3
"""
Test script for the drone swarm simulations
"""

import sys
import os

def test_imports():
    """Test if required packages are available"""
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not available - install with: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib available")
    except ImportError:
        print("✗ Matplotlib not available - install with: pip install matplotlib")
        return False
    
    return True

def main():
    """Main test function"""
    print("Testing Drone Swarm Simulation Requirements...")
    print("=" * 50)
    
    if not test_imports():
        print("\nPlease install required packages:")
        print("pip install numpy matplotlib")
        return
    
    print("\n✓ All required packages available!")
    print("\nAvailable simulations:")
    print("1. drone_swarm_simulation_2d.py - Animated simulation")
    print("2. drone_swarm_static_visualization.py - Static visualization")
    
    print("\nTo run the simulations:")
    print("python drone_swarm_static_visualization.py")
    print("python drone_swarm_simulation_2d.py")

if __name__ == "__main__":
    main()
