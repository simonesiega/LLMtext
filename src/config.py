"""
Configuration file for the mini-GPT project.
"""

import os

# Device: GPU if available, else CPU
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
class Path:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "tiny_shakespeare.txt")
    TOKENIZER_JSON_PATH = os.path.join(BASE_DIR, "data", "tokenizer", "tokenizer.json")

    MODEL_DIR = os.path.join(BASE_DIR, "models")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    TRAIN_BIN = os.path.join(PROCESSED_DIR, "train.bin")
    VAL_BIN = os.path.join(PROCESSED_DIR, "val.bin")
    META_PATH = os.path.join(PROCESSED_DIR, "meta.pkl")

# Tokenizer hyperparameters
class TokenizerConfig:
    MAX_MERGES = 15000
    MIN_FREQ   = 2

# Model hyperparameters
class ModelConfig:
    VOCAB_SIZE = TokenizerConfig.MAX_MERGES + 4
    N_LAYER    = 4
    N_HEAD     = 4
    N_EMBD     = 256
    BLOCK_SIZE = 256
    DROPOUT    = 0.1

# Training hyperparameters
class TrainingConfig:
    BATCH_SIZE = 32
    LR         = 3e-4
    EPOCHS     = 10

# Dataset settings
class DatasetConfig:
    TRAIN_SPLIT = 0.9
    BLOCK_SIZE = 256