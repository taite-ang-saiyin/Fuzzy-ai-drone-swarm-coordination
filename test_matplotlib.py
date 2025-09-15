#!/usr/bin/env python3
"""
Simple test to verify matplotlib works
"""

import numpy as np
import matplotlib.pyplot as plt

def test_basic_plot():
    """Test basic matplotlib functionality"""
    print("Testing matplotlib...")
    
    # Create simple data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Matplotlib Test')
    plt.grid(True)
    plt.legend()
    
    print("✓ Matplotlib plot created successfully!")
    print("Close the plot window to continue...")
    
    plt.show()
    print("✓ Matplotlib test completed!")

if __name__ == "__main__":
    test_basic_plot()
