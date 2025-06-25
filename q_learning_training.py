import numpy as np
import pandas as pd
import pickle
import logging
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTrackQLearningTrainer:
    def __init__(self, track_files=None, q_table_file='multi_track_q_table.pkl', batch_size=20000):
        if track_files is None:
            track_files = ['GL1.csv', 'DL1.csv', 'TL1.csv']  # Default tracks
        self.track_files = track_files
        self.q_table_file = q_table_file
        self.batch_size = batch_size
        self.q_table = defaultdict(lambda: np.zeros(9))  # 9 actions
        self.alpha = 0.1  # Learning rate (reduced for stability)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.8  # High exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slower decay
        self.state_bins = self.define_state_bins()
        self.actions = self.define_actions()
        self.q_table_lock = threading.RLock()  # Thread safety for q_table updates
        self.load_q_table()
        self.state_visit_counts = defaultdict(int)
        self.track_data = {}  # Store data for each track

    def define_state_bins(self):
        """Define bins for discretizing continuous state variables."""
        return {
            'SpeedX': np.linspace(-50, 300, 25),  # Higher resolution for speed
            'TrackPosition': np.linspace(-2.0, 2.0, 10),
            'Angle': np.linspace(-np.pi, np.pi, 12),
            'Track_9': np.linspace(0, 250, 8),
            'Damage': np.linspace(0, 10000, 6),
        }

    def define_actions(self):
        """Define discrete action space: (steer, accel, brake)."""
        steer_options = [-0.5, 0.0, 0.5]
        accel_brake_options = [(1.0, 0.0), (0.5, 0.0), (0.0, 0.0)]  # Favor acceleration
        actions = []
        for steer in steer_options:
            for accel, brake in accel_brake_options:
                actions.append({'steer': steer, 'accel': accel, 'brake': brake})
        return actions

    def discretize_state(self, state):
        """Convert continuous state to discrete state tuple."""
        discrete_state = []
        for key in ['SpeedX', 'TrackPosition', 'Angle', 'Track_9', 'Damage']:
            value = state.get(key, 0.0)
            bins = self.state_bins[key]
            idx = np.digitize(value, bins)
            idx = min(idx, len(bins))  # Ensure index is within bounds
            discrete_state.append(idx)
        return tuple(discrete_state)

    def compute_reward(self, state, next_state):
        """Compute reward with adjusted parameters to prioritize racing objectives."""
        reward = 0.0
        
        # Extract state information
        speed_x = state.get('SpeedX', 0.0)
        next_speed_x = next_state.get('SpeedX', 0.0)
        track_pos = state.get('TrackPosition', 0.0)
        next_track_pos = next_state.get('TrackPosition', 0.0)
        damage = next_state.get('Damage', 0.0) - state.get('Damage', 0.0)
        angle = state.get('Angle', 0.0)
        dist_covered = next_state.get('DistanceCovered', 0.0) - state.get('DistanceCovered', 0.0)
        
        # Speed rewards
        reward += speed_x * 0.1  # Base reward for speed
        reward += (next_speed_x - speed_x) * 5.0  # Reward for acceleration
        
        # Track position rewards
        if abs(next_track_pos) < abs(track_pos):
            reward += 2.0  # Moving toward center
        if abs(track_pos) > 1.0:
            reward -= 5.0  # Penalty for being off track
        
        # Damage penalty (severe)
        if damage > 0:
            reward -= damage * 10.0
        
        # Angle penalty (moderate)
        reward -= abs(angle) * 2.0
        
        # Progress reward (primary objective)
        reward += dist_covered * 10.0
        
        # Catastrophic penalties
        if speed_x < 0:
            reward -= 20.0  # Strong penalty for going backward
        if abs(track_pos) > 1.5:
            reward -= 15.0  # Very strong penalty for being way off track
            
        return reward

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        discrete_state = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.actions))
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula with thread safety."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Use thread lock to ensure thread safety when updating shared resources
        with self.q_table_lock:
            # Increment state visit count
            self.state_visit_counts[discrete_state] += 1
            
            # Adjust learning rate based on visit count for stability
            adjusted_alpha = self.alpha / (1 + 0.1 * self.state_visit_counts[discrete_state])
                        
            # Q-learning update
            current_q = self.q_table[discrete_state][action]
            next_max_q = np.max(self.q_table[discrete_next_state])
            new_q = current_q + adjusted_alpha * (reward + self.gamma * next_max_q - current_q)
            self.q_table[discrete_state][action] = new_q

    def load_q_table(self):
        """Load Q-table from file, ensure defaultdict."""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_table = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(9), loaded_table)
                logger.info(f"Loaded Q-table from file with {len(loaded_table)} states")
            except Exception as e:
                logger.error(f"Error loading Q-table: {e}")

    def save_q_table(self):
        """Save Q-table to file, converting defaultdict to dict with thread safety."""
        try:
            with self.q_table_lock:
                q_table_dict = dict(self.q_table)
            
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(q_table_dict, f)
            logger.info(f"Saved Q-table with {len(q_table_dict)} states to file")
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")

    def analyze_states(self, df, track_name=None):
        """Analyze state distribution to understand coverage."""
        track_info = f" for track {track_name}" if track_name else ""
        logger.info(f"Analyzing state distribution{track_info}...")        
        # Count speed ranges
        speed_bins = [(-50, 0), (0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]
        speed_counts = {}
        for low, high in speed_bins:
            count = len(df[(df['SpeedX'] >= low) & (df['SpeedX'] < high)])
            speed_counts[f"{low}-{high}"] = count
            
        logger.info(f"Speed distribution{track_info}: {speed_counts}")
        
        # Count track position ranges
        pos_bins = [(-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2)]
        pos_counts = {}
        for low, high in pos_bins:
            count = len(df[(df['TrackPosition'] >= low) & (df['TrackPosition'] < high)])
            pos_counts[f"{low}-{high}"] = count
            
        logger.info(f"Track position distribution{track_info}: {pos_counts}")
        
        return speed_counts, pos_counts

    def load_track_thread(self, track_file, results_queue):
        """Load a track's data in a separate thread."""
        try:
            track_name = os.path.splitext(os.path.basename(track_file))[0]
            df = pd.read_csv(track_file)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded {len(df)} records from {track_file}")
            
            # Analyze each track's data distribution
            self.analyze_states(df, track_name)
            
            # Split data into train/validation (only train as we're skipping validation)
            train_df = df  # Use all data for training since we're not evaluating
            
            results_queue.put((track_name, train_df))
        except Exception as e:
            logger.error(f"Error loading track data from {track_file}: {e}")
            results_queue.put((track_name, None))

    def load_track_data(self):
        """Load data from all track CSV files in parallel."""
        results_queue = queue.Queue()
        threads = []
        
        # Start a thread for each track file
        for track_file in self.track_files:
            thread = threading.Thread(
                target=self.load_track_thread,
                args=(track_file, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Process results from queue
        while not results_queue.empty():
            track_name, train_df = results_queue.get()
            if train_df is not None:
                self.track_data[track_name] = {
                    'train': train_df
                }
                logger.info(f"Successfully loaded track {track_name} with {len(train_df)} training records")

    def train_on_track(self, track_name, episode, episodes):
        """Train on a specific track for one episode."""
        train_df = self.track_data[track_name]['train']
        
        # Sample with replacement
        df_sample = train_df.sample(n=min(self.batch_size, len(train_df)), 
                                     random_state=episode).reset_index(drop=True)
        
        # Ensure enough high speed examples are included
        high_speed_df = train_df[train_df['SpeedX'] > 50]
        if len(high_speed_df) > 0:
            high_speed_sample = high_speed_df.sample(n=min(int(self.batch_size * 0.3), 
                                                    len(high_speed_df)), 
                                                    random_state=episode)
            df_sample = pd.concat([df_sample, high_speed_sample]).reset_index(drop=True)
        
        logger.info(f"Episode {episode + 1}/{episodes} on {track_name}: Sampled {len(df_sample)} records")
        total_reward = 0

        for i in range(len(df_sample) - 1):
            state = df_sample.iloc[i].to_dict()
            next_state = df_sample.iloc[i + 1].to_dict()

            action_idx = self.choose_action(state)
            reward = self.compute_reward(state, next_state)
            total_reward += reward

            self.update_q_table(state, action_idx, reward, next_state)

        return total_reward

    def evaluate_on_track(self, track_name, episode):
        """Evaluate model performance on a specific track."""
        val_df = self.track_data[track_name]['val']
        val_rewards = []
        val_sample = val_df.sample(n=min(5000, len(val_df)), random_state=episode)
        
        for i in range(len(val_sample) - 1):
            state = val_sample.iloc[i].to_dict()
            next_state = val_sample.iloc[i + 1].to_dict()
            
            discrete_state = self.discretize_state(state)
            action_idx = np.argmax(self.q_table[discrete_state])
            reward = self.compute_reward(state, next_state)
            val_rewards.append(reward)
        
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0
        logger.info(f"Validation reward on {track_name}: {avg_val_reward:.2f}")
        
        return avg_val_reward

    def train_track_thread(self, track_name, episode, episodes, results_queue):
        """Train on a track in a separate thread and put results in queue."""
        try:
            total_reward = self.train_on_track(track_name, episode, episodes)
            results_queue.put((track_name, total_reward))
        except Exception as e:
            logger.error(f"Error in thread for track {track_name}: {str(e)}")
            results_queue.put((track_name, 0.0))  # Return 0 reward on error

    def train(self, episodes=100, max_workers=3):
        """Train Q-learning model using data from multiple tracks with multithreading."""
        logger.info(f"Training on tracks: {', '.join(self.track_files)}")
        
        # Load data from all tracks
        self.load_track_data()
        
        if not self.track_data:
            logger.error("No track data loaded. Aborting training.")
            return
        
        # Use the minimum of available tracks and requested workers
        num_workers = min(len(self.track_data), max_workers)
        logger.info(f"Training with {num_workers} parallel workers")
        
        # Save q-table periodically (every 20 episodes)
        save_interval = 20
        
        for episode in range(episodes):
            # Use ThreadPoolExecutor for parallel processing of tracks
            results_queue = queue.Queue()
            threads = []
            
            for track_name in self.track_data.keys():
                thread = threading.Thread(
                    target=self.train_track_thread,
                    args=(track_name, episode, episodes, results_queue)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Process results
            episode_rewards = {}
            while not results_queue.empty():
                track_name, reward = results_queue.get()
                episode_rewards[track_name] = reward
                logger.info(f"Episode {episode + 1}/{episodes}, Track {track_name}, "
                          f"Total Reward: {reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            # Decay epsilon once per episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Save periodically without evaluation overhead
            if (episode + 1) % save_interval == 0:
                logger.info(f"Episode {episode + 1}/{episodes} completed. Saving Q-table...")
                self.save_q_table()
        
        # Print final statistics
        non_zero_states = sum(1 for values in self.q_table.values() if np.max(values) > 0)
        logger.info(f"Training complete. Q-table has {non_zero_states} non-zero states")
        logger.info(f"Most visited states: {sorted(self.state_visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # Final save
        self.save_q_table()

if __name__ == "__main__":
    # List of track CSV files to train on
    track_files = ['GL.csv', 'DL.csv', 'TL.csv']
    
    # Create a trainer with the three tracks
    trainer = MultiTrackQLearningTrainer(track_files=track_files)
    
    # Train with multithreading (one thread per track)
    trainer.train(episodes=100, max_workers=3)  # Increased episodes since we removed evaluation overhead