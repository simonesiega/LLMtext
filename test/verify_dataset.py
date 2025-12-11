import os
import numpy as np
import torch
import sys

# Add src to Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
sys.path.append(os.path.join(base_dir, "src"))

# Import Rust-based tokenizer wrapper
from tokens.subword_tokenizer_rust import SubwordTokenizerRust

# Import configuration module
from config import TOKENIZER_JSON_PATH, TRAIN_BIN, VAL_BIN

# Number of tokens to display for verification
N = 50  

def load_bin(path):
    """
    Load a binary file containing token IDs as unsigned 16-bit integers
    and convert it into a PyTorch LongTensor.

    Parameters
    ----------
    path : str
        Path to the .bin file to load.

    Returns
    -------
    torch.LongTensor
        Tensor containing token IDs.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    """

    print(f"Loading binary file: {path}")

    # Load binary data as NumPy array of uint16
    data = np.fromfile(path, dtype=np.uint16)

    # Convert NumPy array to PyTorch LongTensor
    data = torch.from_numpy(data).long()   
    return data


def decode_tokens(tok, ids):
    """
    Decode a sequence of token IDs back to a string using the Rust tokenizer.

    Parameters
    ----------
    tok : SubwordTokenizerRust
        Instance of the Rust tokenizer wrapper.
    ids : torch.LongTensor
        Tensor of token IDs to decode.

    Returns
    -------
    str
        Decoded text string.
    """
    # Convert tensor to Python list before passing to Rust decoder
    ids = ids.tolist()
    return tok.decode_from_ids(ids)


def main():
    """
    Main function to preprocess the dataset for LLM training.
    """
    # Load tokenizer
    tok = SubwordTokenizerRust.load(TOKENIZER_JSON_PATH)
    print(f"Loaded tokenizer with vocab size: {len(tok.id_to_token)}")

    # Load train and validation datasets
    train_ids = load_bin(TRAIN_BIN)
    val_ids = load_bin(VAL_BIN)

    print(f"Train tokens: {len(train_ids)}")
    print(f"Val   tokens: {len(val_ids)}")

    # Show first N tokens
    print("\nFirst tokens (train):")
    print(train_ids[:N].tolist())

    # Decode first N tokens to text
    try:
        print("\nDecoded snippet:\n")
        text = decode_tokens(tok, train_ids[:N])
        print(text)
    except Exception as e:
        print("Could not decode tokens:", e)

    # Dataset length sanity check
    if len(train_ids) < 100:
        print("WARNING: train set extremely small.")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
