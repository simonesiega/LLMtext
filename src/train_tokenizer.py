import os
import sys
import time

# Add src to Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(base_dir, "src"))

# Import the Rust tokenizer wrapper 
from tokens.subword_tokenizer_rust import SubwordTokenizerRust
from config import RAW_DATA, TOKENIZER_SAVE, MAX_MERGES, MIN_FREQ


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
    print(f"Reading dataset from: {RAW_DATA}")
    text = read_text(RAW_DATA)

    # Initialize the tokenizer (calls Rust SubwordTokenizer::new())
    tok = SubwordTokenizerRust()
    
    # Start timer
    start = time.time()

    tok.train([text], max_merges=MAX_MERGES, min_freq_token=MIN_FREQ)

    # Stop timer
    end = time.time()
    print(f"Done in {end - start:.2f}s.")

    # Debugging stats
    # print(f"Vocabulary size: {len(tok.id_to_token)}")
    # print(f"Number of merges: {len(tok.merges)}")
    # print(f"Special tokens: {tok.special_tokens}")

    # Save the trained tokenizer to JSON
    tok.save(TOKENIZER_SAVE)
    print(f"Tokenizer saved to: {TOKENIZER_SAVE}")


if __name__ == "__main__":
    main()
