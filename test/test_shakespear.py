import sys
import time
import os

# Add src to Python path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
sys.path.append(os.path.join(base_dir, "src"))

# Import the Rust subword tokenizer wrapper 
from tokens.subword_tokenizer_rust import SubwordTokenizerRust

# Function to read text data from a file 
def read_data(path):
    # Read the file line by line, strip whitespace, and skip empty lines
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    # Build the path to the tiny_shakespeare.txt file relative to the script 
    file_path = os.path.join(base_dir, "data_test", "tiny_shakespeare.txt")
    
    # Load all text lines 
    texts = read_data(file_path)
    print(f"Number of lines read: {len(texts)}")

    # Initialize the tokenizer 
    tok = SubwordTokenizerRust()

    # Set training parameters 
    vocab_size = 15000 # maximum number of merges/tokens
    min_freq = 2 # minimum frequency to consider a merge

    # Train the tokenizer 
    start_time = time.time()
    tok.train(texts, vocab_size, min_freq)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Save the tokenizer to a JSON file 
    tok.save("tokenizer_shakespeare.json")
    print("\nTokenizer saved successfully.")

    print(f"Number of tokens saved: {len(tok.merges)}")

# Main
if __name__ == "__main__":
    main()