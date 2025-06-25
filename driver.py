#!/usr/bin/env python3
import pickle
import numpy as np
import math
import logging
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QLearningDriver:
    def __init__(self, stage):
        self.stage = stage
        self.q_table_file = 'q_table.pkl'
        self.q_table = defaultdict(lambda: np.zeros(15))  # 15 actions
        self.state_bins = self.define_state_bins()
        self.actions = self.define_actions()
        self.epsilon = 0.1
        self.load_q_table()
        self.prev_speedX = 0
        self.prev_state = None
        self.prev_steer = 0.0  # For damping
        self.gear_change_delay = 0
        self.stateCounter = {}
        self.current_gear = 1
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_steps = 0
        self.track_history = []  # Store recent track positions to detect oscillation
        
        # Track centering parameters - ADJUSTED
        self.centering_weight = 0.10  # Reduced for less aggressive steering
        self.angle_weight = 0.5  # More balanced
        self.damping_factor = 0.5  # Moderate damping

    def init(self):
        return "(init racer)"

    def define_state_bins(self):
        return {
            'SpeedX': np.linspace(-50, 300, 25),
            'TrackPosition': np.linspace(-2.0, 2.0, 10),
            'Angle': np.linspace(-np.pi, np.pi, 20),  # Finer bins
            'Track_9': np.linspace(0, 250, 8),
            'Damage': np.linspace(0, 10000, 6),
        }

    def define_actions(self):
        steer_options = [-0.5, -0.25, 0.0, 0.25, 0.5]  # Finer steering
        accel_brake_options = [(1.0, 0.0), (0.5, 0.0), (0.0, 0.0)]
        actions = []
        for steer in steer_options:
            for accel, brake in accel_brake_options:
                actions.append({'steer': steer, 'accel': accel, 'brake': brake})
        return actions

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(15), loaded_table)
                logger.info(f"Loaded Q-table with {len(loaded_table)} states")
                non_zero_states = sum(1 for values in self.q_table.values() if np.max(values) > 0)
                logger.info(f"Q-table has {non_zero_states} non-zero states")
                sample_states = list(loaded_table.keys())[:5]
                for state in sample_states:
                    logger.info(f"Sample state {state}: actions {self.q_table[state]}")
            except Exception as e:
                logger.error(f"Error loading Q-table: {e}")
                logger.info("Using new Q-table")

    def onShutDown(self):
        logger.info("Shutting down")
        top_states = sorted(self.stateCounter.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top states encountered: {top_states}")

    def onRestart(self):
        logger.info("Restarting")
        self.prev_state = None
        self.prev_steer = 0.0
        self.gear_change_delay = 0
        self.current_gear = 1
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_steps = 0
        self.track_history = []

    def get_state_from_string(self, string):
        d = {}
        start = string.find('angle') + 6
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['Angle'] = float(string[start:end])
        start = string.find('speedX') + 7
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['SpeedX'] = float(string[start:end])
        start = string.find('trackPos') + 9
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['TrackPosition'] = float(string[start:end])
        start = string.find('track') + 6
        end = string.find(')', start)
        if start != -1 and end != -1:
            track_str = string[start:end].strip()
            track_values = [float(v) for v in track_str.split()]
            if len(track_values) >= 19:
                d['Track_9'] = track_values[8]
                d['TrackFront'] = track_values[8:12]
                d['TrackAll'] = track_values
        start = string.find('damage') + 7
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['Damage'] = float(string[start:end])
        start = string.find('gear') + 5
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['Gear'] = int(string[start:end])
        start = string.find('rpm') + 4
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['RPM'] = float(string[start:end])
        start = string.find('distRaced') + 10
        end = string.find(')', start)
        if start != -1 and end != -1:
            d['DistanceCovered'] = float(string[start:end])
        return d

    def discretize_state(self, state):
        discrete_state = []
        for key in ['SpeedX', 'TrackPosition', 'Angle', 'Track_9', 'Damage']:
            value = state.get(key, 0.0)
            bins = self.state_bins[key]
            idx = np.digitize(value, bins)
            idx = min(idx, len(bins))
            discrete_state.append(idx)
        return tuple(discrete_state)

    def detect_stuck(self, state):
        speedX = state.get('SpeedX', 0.0)
        if speedX < 3.0:  # CHANGED: Lower threshold to detect truly stuck situations
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)  # CHANGED: Faster reset
        if self.stuck_counter > 25:  # CHANGED: Require longer time to confirm stuck
            if not self.recovery_mode:
                logger.warning("Car stuck. Entering recovery mode.")
                self.recovery_mode = True
                self.recovery_steps = 30
            return True
        return False

    def recover_from_stuck(self):
        self.recovery_steps -= 1
        if self.recovery_steps <= 0:
            self.recovery_mode = False
            self.stuck_counter = 0
            logger.info("Exiting recovery mode")
            return 0.0, 1.0, 0.0, 1
        if self.recovery_steps > 20:
            return -0.3, 0.0, 0, -1  # CHANGED: Less braking, more steering
        else:
            return 0.3, 1.0, 0.0, 1  # CHANGED: Full acceleration for recovery

    def detect_oscillation(self, trackPos):
        # Store track positions to detect oscillation
        self.track_history.append(trackPos)
        if len(self.track_history) > 10:
            self.track_history.pop(0)
            
        # Check if car is oscillating across the track
        if len(self.track_history) >= 6:
            crossings = 0
            for i in range(1, len(self.track_history)):
                if (self.track_history[i-1] < 0 and self.track_history[i] > 0) or \
                   (self.track_history[i-1] > 0 and self.track_history[i] < 0):
                    crossings += 1
            
            # If many crossings, increase damping temporarily
            if crossings >= 4:  # CHANGED: Higher threshold
                logger.warning("Oscillation detected, increasing damping")
                return True
        return False

    def apply_safety_checks(self, steer, accel, brake, track_sensors, speed):
        # CHANGED: Less conservative safety checks
        min_front_dist = min(track_sensors[:3]) if track_sensors else 100
        
        # Progressive braking based on front distance and speed
        if min_front_dist < 7:  # CHANGED: Lower threshold
            brake_factor = max(0, (7 - min_front_dist) / 7)
            brake = max(brake, brake_factor * (0.4 + min(0.3, speed/150)))  # CHANGED: Less brake
            accel = 0.0
        elif min_front_dist < 20:  # CHANGED: Lower threshold
            # Reduce acceleration when approaching obstacles
            accel_factor = min_front_dist / 20
            accel = min(accel, accel_factor * 0.9)  # CHANGED: Higher factor
        
        return steer, accel, brake

    def rule_based_control(self, state):
        angle = state.get('Angle', 0.0)
        trackPos = state.get('TrackPosition', 0.0)
        speedX = state.get('SpeedX', 0.0)
        track_sensors = state.get('TrackFront', [100, 100, 100, 100])
        all_track = state.get('TrackAll', [100] * 19)

        # CHANGED: Curve detection - less sensitive thresholds
        track_diff = max([abs(all_track[i] - all_track[18-i]) for i in range(9)])
        left_side_avg = sum(all_track[:9]) / 9
        right_side_avg = sum(all_track[10:]) / 9
        side_diff = abs(left_side_avg - right_side_avg)
        
        # CHANGED: Higher thresholds for curve detection
        in_curve = track_diff > 25 or side_diff > 15
        sharp_curve = track_diff > 40 or side_diff > 30

        # Check for oscillation
        oscillating = self.detect_oscillation(trackPos)
        
        # Steering calculation with adaptive weights
        # 1. Angle correction (keep car pointed in right direction)
        steering_angle = angle * self.angle_weight
        
        # 2. Track position correction (keep car centered)
        # Use progressive centering - more aggressive when further from center
        if abs(trackPos) > 0.7:  # CHANGED: Only aggressive when far from center
            # Far from center - stronger correction
            centering = -trackPos * (self.centering_weight * 1.5)
        else:
            # Near center - gentler correction
            centering = -trackPos * self.centering_weight
            
        # 3. Curve handling adjustment
        if in_curve:
            # In curves, prioritize angle over centering
            steering_angle *= 1.2  # Increase angle weight
            centering *= 0.8       # Reduce centering weight
            
            # Look ahead for curve direction
            if left_side_avg > right_side_avg:
                # Right curve coming - bias slightly right
                centering -= 0.05
            elif right_side_avg > left_side_avg:
                # Left curve coming - bias slightly left
                centering += 0.05
                
        # Combine steering components
        steer = steering_angle + centering
        
        # Apply additional damping if oscillating
        damping = self.damping_factor
        if oscillating:
            damping = 0.8  # Stronger damping when oscillating
            
        # Apply damping (smoothing with previous steering)
        steer = (1 - damping) * steer + damping * self.prev_steer
        
        # Limit steering to valid range
        steer = max(-1.0, min(1.0, steer))

        # CHANGED: Speed control - higher speeds
        accel = 1.0  # Default to full acceleration
        brake = 0.0
        
        # CHANGED: Higher target speeds
        if sharp_curve:
            target_speed = 90  # Faster for sharp curves (was 60)
        elif in_curve:
            target_speed = 140  # Faster for normal curves (was 100)
        else:
            target_speed = 200  # Faster on straights (was 150)
            
        # CHANGED: Less conservative speed control
        if speedX > target_speed * 1.2:  # Only brake when well over target
            accel = 0.2
            brake = 0.05  # Lighter braking
        elif speedX > target_speed:
            accel = 0.6  # Maintain higher speed (was 0.3)
        
        # CHANGED: Less conservative safety checks
        min_front_dist = min(track_sensors)
        if min_front_dist < 15:  # Lower threshold (was 20)
            accel = 0.7 * (min_front_dist / 15)  # Higher factor
        if min_front_dist < 8:  # Lower threshold (was 10)
            accel = 0.0
            brake = 0.1 + (8 - min_front_dist) * 0.05  # Less braking
        if min_front_dist < 3:  # Lower threshold (was 5)
            brake = 0.5  # Less braking (was 0.8)
            
        # CHANGED: Optimized gear shifting logic
        rpm = state.get('RPM', 0)
        gear = self.current_gear
        if self.gear_change_delay > 0:
            self.gear_change_delay -= 1
        else:
            if rpm > 8500 and gear < 6:  # Higher RPM threshold (was 8000)
                gear += 1
                self.gear_change_delay = 3  # Shorter delay (was 5)
            elif rpm < 3000 and gear > 1:
                gear -= 1
                self.gear_change_delay = 3  # Shorter delay (was 5)
            if speedX < 5 and gear < 1:
                gear = 1
        self.current_gear = gear

        logger.info(f"Rule-based: Angle={angle:.2f}, TrackPos={trackPos:.2f}, Steer={steer:.2f}, " +
                    f"PrevSteer={self.prev_steer:.2f}, Accel={accel:.2f}, " +
                    f"InCurve={in_curve}, Oscillating={oscillating}")
        
        self.prev_steer = steer
        return steer, accel, brake, gear

    def drive(self, string):
        current_state = self.get_state_from_string(string)
        speedX = current_state.get('SpeedX', 0.0)
        trackPos = current_state.get('TrackPosition', 0.0)

        if self.detect_stuck(current_state):
            if self.recovery_mode:
                steer, accel, brake, gear = self.recover_from_stuck()
                control = f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)(focus 0)(meta 0)"
                return control

        discrete_state = self.discretize_state(current_state)
        if discrete_state not in self.stateCounter:
            self.stateCounter[discrete_state] = 0
        self.stateCounter[discrete_state] += 1

        track_sensors = current_state.get('TrackFront', [100, 100, 100, 100])

        # CHANGED: Lower threshold for using Q-table
        if np.max(self.q_table[discrete_state]) > 0.3:  # Lower threshold for using Q-table (was 0.5)
            if np.random.random() < self.epsilon:
                action_idx = np.random.choice(len(self.actions))
            else:
                action_idx = np.argmax(self.q_table[discrete_state])
            action = self.actions[action_idx]
            steer = action['steer']
            accel = action['accel']
            brake = action['brake']
            
            # CHANGED: Less damping for Q-learning steering
            steer = 0.4 * steer + 0.6 * self.prev_steer  # Was 0.3/0.7
            
            # Safety layer
            steer, accel, brake = self.apply_safety_checks(steer, accel, brake, track_sensors, speedX)

            rpm = current_state.get('RPM', 0)
            gear = self.current_gear
            if self.gear_change_delay > 0:
                self.gear_change_delay -= 1
            else:
                if rpm > 8500 and gear < 6:  # Higher RPM threshold (was 8000)
                    gear += 1
                    self.gear_change_delay = 3  # Shorter delay (was 5)
                elif rpm < 3000 and gear > 1:
                    gear -= 1
                    self.gear_change_delay = 3  # Shorter delay (was 5)
                if speedX < 5 and gear < 1:
                    gear = 1
            self.current_gear = gear

            logger.info(f"Q-table: State={discrete_state}, Action={action_idx}, " +
                        f"Steer={steer:.2f}, Accel={accel:.2f}, TrackPos={trackPos:.2f}")
        else:
            # Fall back to improved rule-based control
            steer, accel, brake, gear = self.rule_based_control(current_state)

        self.prev_state = current_state
        self.prev_speedX = speedX
        self.prev_steer = steer  # Update previous steering angle
        
        control = f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)(focus 0)(meta 0)"
        return control