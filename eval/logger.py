import csv, os
from datetime import datetime

class Logger:
    def __init__(self, log_dir, drone_id):
        os.makedirs(log_dir, exist_ok=True)
        self.file = open(os.path.join(log_dir, f"drone_{drone_id}.csv"), 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp','step','drone_id','x','y','z','vx','vy','vz','yaw','d_obstacle','neighbor_count','neighbor_avg_dist','thrust_adj','yaw_rate','collided','goal_dx','goal_dy','goal_dz','reward'])
    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()
    def close(self):
        self.file.close()
