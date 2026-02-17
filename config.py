
import os

# Paths

DATA_DIR = os.environ.get("OOD_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OOD_OUTPUT_DIR", "./outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Reproducibility

SEED = 42

# Data

NUM_CLASSES = 100  # CIFAR-100

# CIFAR-100 channel means and stds 
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

BATCH_SIZE = 128
NUM_WORKERS = 2  # parallel data loading 

# Training

NUM_EPOCHS = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_MIN = 1e-6  

#  checkpoints 
CHECKPOINT_EVERY = 10
# training progress 
PRINT_EVERY = 10

# OOD

ENERGY_TEMPERATURE = 1.0  # temperature for the energy score

# Helper


def ensure_dirs():
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
