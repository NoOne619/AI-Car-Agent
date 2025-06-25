import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 20000

# Input features
input_features = [
    'angle', 'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
    'track_-90', 'track_-75', 'track_-60', 'track_-45', 'track_-30', 'track_-20', 
    'track_-15', 'track_-10', 'track_-5', 'track_0', 'track_5', 'track_10', 
    'track_15', 'track_20', 'track_30', 'track_45', 'track_60', 'track_75', 'track_90',
    'opponent_0', 'opponent_1', 'opponent_2', 'opponent_3', 'opponent_4', 
    'opponent_5', 'opponent_6', 'opponent_7'
]

# Output features
output_features = ['accel', 'brake', 'steer']

# Initialize data
data = {feat: np.zeros(n_samples) for feat in input_features + output_features}

# Generate improved synthetic telemetry and controls
for i in range(n_samples):
    # Telemetry: Mimic varied TORCS sensor data
    data['angle'][i] = np.random.uniform(-0.7, 0.7)  # Wider angle range
    data['speedX'][i] = np.random.uniform(0, 200)  # Wider speed range
    data['speedY'][i] = np.random.uniform(-30, 30)  # Wider sideways speed
    data['speedZ'][i] = np.random.uniform(-15, 15)  # Wider vertical speed
    data['trackPos'][i] = np.random.uniform(-1.5, 1.5)  # Wider track position
    data['z'][i] = np.random.uniform(0, 1.0)  # Wider height range
    
    # Track sensors: Simulate turns, obstacles, and clear paths
    is_turn = np.random.random() < 0.4  # 40% chance of turn/obstacle
    is_obstacle = np.random.random() < 0.2  # 20% chance of close obstacle
    for angle in [-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90]:
        if is_obstacle and abs(angle) < 30:  # Close obstacle
            data[f'track_{angle}'][i] = np.random.uniform(5, 20)
        elif is_turn and abs(angle) < 45:  # Turn
            data[f'track_{angle}'][i] = np.random.uniform(20, 100)
        else:  # Clear path
            data[f'track_{angle}'][i] = np.random.uniform(100, 200)
    
    # Opponent distances: Closer opponents for competition
    for j in range(8):
        proximity = np.random.random()
        data[f'opponent_{j}'][i] = np.random.uniform(10, 200) if proximity < 0.3 else np.random.uniform(100, 200)
    
    # Heuristic controls with smoother transitions
    track_center = data['track_0'][i]
    speed_x = data['speedX'][i]
    angle = data['angle'][i]
    track_pos = data['trackPos'][i]
    
    # Steering: Smoother adjustment
    steer = -angle / 0.785398 * 0.5  # Reduced sensitivity
    steer -= track_pos * 0.3  # Adjusted influence
    data['steer'][i] = np.clip(steer, -1.0, 1.0)
    
    # Acceleration/Braking: Dynamic based on speed and track
    if track_center < 20.0:  # Close obstacle
        data['accel'][i] = 0.0
        data['brake'][i] = np.clip(0.7 * (speed_x / 100), 0.0, 1.0)  # Brake proportional to speed
    elif track_center < 50.0:  # Turn
        accel_factor = max(0, 1.0 - (50 - track_center) / 30)
        data['accel'][i] = np.clip(accel_factor, 0.0, 1.0)
        data['brake'][i] = 0.0 if speed_x < 150 else np.clip((speed_x - 150) / 50, 0.0, 1.0)
    else:  # Clear path
        data['accel'][i] = np.clip(1.0 - speed_x / 200, 0.0, 1.0)  # Gradual acceleration
        data['brake'][i] = 0.0

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('telemetricData.csv', index=False)
print("Improved telemetricData.csv generated successfully with 20,000 samples.")