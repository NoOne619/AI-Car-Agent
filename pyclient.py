#!/usr/bin/env python3
import sys
import argparse
import socket
import driver
import csv
import os
import re
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    logger.error(f'Could not make a socket: {msg}')
    sys.exit(-1)

# Three second timeout
sock.settimeout(3.0)

shutdownClient = False
curEpisode = 0
verbose = True  # Enable verbose logging

# Initialize QLearningDriver
d = driver.QLearningDriver(arguments.stage)

# Define CSV headers (74 input features + 5 control outputs)
headers = [
    'Angle', 'CurrentLapTime', 'Damage', 'DistanceFromStart', 'DistanceCovered', 'FuelLevel',
    'Gear', 'LastLapTime', 'RacePosition', 'RPM', 'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPosition', 'Z',
    *[f'Opponent_{i}' for i in range(1, 37)],
    *[f'Track_{i}' for i in range(1, 20)],
    *[f'WheelSpinVelocity_{i}' for i in range(1, 5)],
    'Acceleration', 'Braking', 'Clutch', 'Gear_Control', 'Steering'
]

# Initialize CSV
if not os.path.exists('TL1.csv'):
    with open('TL1.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
csv_file = open('TL1.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# Parse function for both received and control strings
def parse_data(buf):
    data = {}
    matches = re.findall(r'\(([^)]+)\)', buf)
    for match in matches:
        parts = match.split()
        key = parts[0]
        values = parts[1:]
        if len(values) == 1:
            try:
                data[key] = float(values[0]) if '.' in values[0] else int(values[0])
            except ValueError:
                data[key] = values[0]
        else:
            data[key] = [float(v) if '.' in v else int(v) for v in values]
    return data

while not shutdownClient:
    retries = 5  # Increased retries
    while retries > 0:
        logger.info(f'Sending id to server: {arguments.id}')
        buf = arguments.id + d.init()
        logger.info(f'Sending init string to server: {buf}')
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            logger.error(f"Failed to send data: {msg}")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
            logger.info(f'Received: {buf}')
            if '***identified***' in buf:
                break
        except socket.error as msg:
            logger.warning(f"Didn't get response from server, retries left: {retries}")
            retries -= 1
            time.sleep(2)  # Longer delay
        if retries == 0:
            logger.error("Failed to connect to server after retries")
            sys.exit(-1)

    currentStep = 0
    
    while True:
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
            if verbose:
                logger.info(f'Received: {buf}')
        except socket.error as msg:
            logger.warning("Didn't get response from server, skipping...")
            continue
        
        if '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            logger.info('Client Shutdown')
            break
        
        if '***restart***' in buf:
            d.onRestart()
            logger.info('Client Restart')
            break
        
        # Parse received data
        try:
            received_data = parse_data(buf)
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            continue
        
        # Get control data
        currentStep += 1
        control_buf = None
        if currentStep != arguments.max_steps:
            try:
                control_buf = d.drive(buf)
            except Exception as e:
                logger.error(f"Error in drive: {e}")
                continue
        else:
            control_buf = '(meta 1)'
        
        try:
            control_data = parse_data(control_buf)
        except Exception as e:
            logger.error(f"Error parsing control data: {e}")
            continue
        
        if verbose:
            logger.info(f'Sending: {control_buf}')
        
        # Construct CSV row
        row = (
            [
                received_data.get('angle', 0.0),
                received_data.get('curLapTime', 0.0),
                received_data.get('damage', 0.0),
                received_data.get('distFromStart', 0.0),
                received_data.get('distRaced', 0.0),
                received_data.get('fuel', 0.0),
                received_data.get('gear', 0),
                received_data.get('lastLapTime', 0.0),
                received_data.get('racePos', 0),
                received_data.get('rpm', 0.0),
                received_data.get('speedX', 0.0),
                received_data.get('speedY', 0.0),
                received_data.get('speedZ', 0.0),
                received_data.get('trackPos', 0.0),
                received_data.get('z', 0.0),
            ] +
            (received_data.get('opponents', [200.0] * 36) + [200.0] * 36)[:36] +
            (received_data.get('track', [200.0] * 19) + [200.0] * 19)[:19] +
            (received_data.get('wheelSpinVel', [0.0] * 4) + [0.0] * 4)[:4] +
            [
                control_data.get('accel', 0.0),
                control_data.get('brake', 0.0),
                control_data.get('clutch', 0.0),
                control_data.get('gear', 0),
                control_data.get('steer', 0.0)
            ]
        )
        try:
            csv_writer.writerow(row)
            csv_file.flush()
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")
        
        try:
            sock.sendto(control_buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            logger.error(f"Failed to send data: {msg}")
            sys.exit(-1)
    
    curEpisode += 1
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()
csv_file.close()