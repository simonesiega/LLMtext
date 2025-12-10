"""
Configuration file for the mini-GPT project.
"""

import os

# Device
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "tiny_shakespeare.txt")
TOKENIZER_SAVE = os.path.join(BASE_DIR, "data", "tokenizer", "tokenizer.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Tokenizer hyperparameters
MAX_MERGES = 15000       
MIN_FREQ   = 2           

# Model hyperparameters
VOCAB_SIZE = MAX_MERGES + 4  
N_LAYER    = 4               
N_HEAD     = 4            
N_EMBD     = 256            
BLOCK_SIZE = 256            
DROPOUT    = 0.1

# Training hyperparameters
BATCH_SIZE = 32
LR         = 3e-4
EPOCHS     = 10
