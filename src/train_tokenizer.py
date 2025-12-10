import os
import sys
import time

# Add src to Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(base_dir, "src"))

# Import the Rust tokenizer wrapper 
from tokens.subword_tokenizer_rust import SubwordTokenizerRust


def read_text(path):
    """
    Read the entire file as a single string.

    Parameters
    ----------
    path : str
        Path to the text file to load.

    Returns
    -------
    str
        Full contents of the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    """
    Train the Rust-based tokenizer on the raw dataset and save it to JSON.
    """

    # Construct paths relative to project root
    raw_path  = os.path.join(base_dir, "data", "raw", "tiny_shakespeare.txt")
    save_path = os.path.join(base_dir, "data", "tokenizer", "tokenizer.json")

    print(f"Reading dataset from: {raw_path}")
    text = read_text(raw_path)

    # Initialize the tokenizer (calls Rust SubwordTokenizer::new())
    tok = SubwordTokenizerRust()

    # Training hyperparameters:
    # max_merges defines how many merge operations BPE will perform,
    # effectively setting the vocabulary size.
    # min_freq ensures merges with extremely low frequency are ignored.
    max_merges = 15000
    min_freq   = 2
    
    # Start timer
    start = time.time()

    tok.train([text], max_merges=max_merges, min_freq_token=min_freq)

    # Stop timer
    end = time.time()
    print(f"Done in {end - start:.2f}s.")

    # Debugging stats
    # print(f"Vocabulary size: {len(tok.id_to_token)}")
    # print(f"Number of merges: {len(tok.merges)}")
    # print(f"Special tokens: {tok.special_tokens}")

    # Save the trained tokenizer to JSON
    tok.save(save_path)
    print(f"Tokenizer saved to: {save_path}")


if __name__ == "__main__":
    main()
