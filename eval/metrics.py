import pandas as pd
import numpy as np

def collisions(df):
    return int(df['collided'].sum())

def avg_cohesion(df):
    return float(df['neighbor_avg_dist'].mean())

def coverage(df, cell_size=2.0):
    x = (df['x'] // cell_size).astype(int)
    y = (df['y'] // cell_size).astype(int)
    z = (df['z'] // cell_size).astype(int)
    return len(set(zip(x, y, z)))

def episode_summary(df):
    return {
        'collisions': collisions(df),
        'avg_cohesion': avg_cohesion(df),
        'coverage': coverage(df),
        'reward_sum': float(df['reward'].sum()),
        'steps': int(df['step'].max()),
    }
