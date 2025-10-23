"""
Configuration file for network scheduling simulation
"""

# Network parameters
EDCA_PREAMBLE_OVERHEAD = 150e-6  # 150 microseconds in seconds
MCS11_RATE = 2442e6  # 2442 Mbps in bits per second
MAX_TRANSMISSION_TIME = 4e-3  # 4 milliseconds in seconds

# User and service parameters
MIN_USERS = 1
MAX_USERS = 64
SERVICE_TYPES = ['VO', 'VI', 'BE']

# Delay thresholds (in seconds)
VO_DELAY_THRESHOLD = 20e-3  # 20ms
VI_DELAY_THRESHOLD_PRIORITY = 20e-3  # 20ms for priority scheduling
VI_DELAY_THRESHOLD_DROP = 50e-3  # 50ms for dropping
BE_MIN_INTERVAL = 100e-3  # 100ms minimum service interval

# Simulation parameters
SIMULATION_TIME_STEP = 1e-3  # 1ms simulation tick
PACKET_SIZE_RANGE = (500, 1500)  # Packet size in bytes

# Traffic model parameters
PERIODIC_INTERVAL_RANGE = (5e-3, 50e-3)  # 5-50ms for periodic traffic
POISSON_LAMBDA_RANGE = (10, 100)  # Average packets per second

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
TRAIN_TEST_SPLIT = 0.8
