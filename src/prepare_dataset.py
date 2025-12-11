import os
import sys
import pickle
import numpy as np
import time

# Add src to Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(base_dir, "src"))

# Import Rust-based tokenizer wrapper
from tokens.subword_tokenizer_rust import SubwordTokenizerRust

# Import configuration module
from config import (
    RAW_DATA_PATH, # Path to raw text file
    TOKENIZER_JSON_PATH,  # Path to saved tokenizer JSON
    PROCESSED_DIR, # Directory to save processed data
    TRAIN_BIN, # Path to save train token IDs binary
    VAL_BIN, # Path to save validation token IDs binary
    META_PATH, # Path to save metadata (vocab etc.)
    TRAIN_SPLIT # Fraction of data to use as training set
)


def read_text(path):
    """
    Read the entire text file as a single string.
    
    Parameters
    ----------
    path : str
        Path to the raw text file.
    
    Returns
    -------
    str
        The full text content of the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    """
    Main function to preprocess the dataset for LLM training.
    """
    # Start global timer
    start_total = time.time()


    # 1: Load raw text
    print(f"1 Loading raw text from: {RAW_DATA_PATH}")
    t0 = time.time()
    text = read_text(RAW_DATA_PATH)
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds. Total characters read: {len(text):,}")

    # 2: Load tokenizer
    print("2 Loading tokenizer from JSON")
    t0 = time.time()
    tok = SubwordTokenizerRust.load(TOKENIZER_JSON_PATH)
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds. Vocabulary size: {len(tok.id_to_token)}")

    # 3: Encode full dataset
    print("3 Tokenizing full dataset into IDs")
    t0 = time.time()
    # Convert text to list of integer token IDs
    ids = tok.encode_ids(text)  
    # Use uint16 for vocab ≤ 65535
    ids = np.array(ids, dtype=np.uint16)  
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds. Total tokens: {len(ids):,}")

    # Create processed directory if it does not exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 5: Split dataset into train/validation
    print("4 Splitting dataset into training and validation sets")
    t0 = time.time()
    split_idx = int(len(ids) * TRAIN_SPLIT)
    train_ids = ids[:split_idx]
    val_ids   = ids[split_idx:]
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds.")
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    # 5: Save binary token ID files
    print("5 Saving binary token ID files")
    t0 = time.time()
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds.")
    print(f"Saved train.bin → {TRAIN_BIN}")
    print(f"Saved val.bin   → {VAL_BIN}")

    # 6: Save tokenizer metadata 
    print("6 Saving tokenizer metadata")
    t0 = time.time()
    meta = {
        "vocab_size": len(tok.id_to_token),
        "token_to_id": tok.token_to_id,
        "id_to_token": tok.id_to_token,
        "special_tokens": tok.special_tokens,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f} seconds. Metadata saved to: {META_PATH}")

    # Total preprocessing time
    end_total = time.time()
    print(f"\nTotal preprocessing completed in {end_total - start_total:.2f} seconds.")


if __name__ == "__main__":
    main()
