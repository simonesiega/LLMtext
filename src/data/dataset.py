import os
import numpy as np
import torch

# Import configuration module
from config import (
    TRAIN_BIN,
    VAL_BIN,
    BLOCK_SIZE,
    BATCH_SIZE,
    device
)


class GPTDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset implementing the Next-Token Prediction (NTP) objective.

    Next-Token Prediction:
    Given a sequence of tokens:

        [t1, t2, t3, t4, ..., tN]

    The model learns to predict the next token at every position:

        input : [t1, t2, t3, ..., t(N-1)]
        target: [t2, t3, t4, ..., tN]

    So at training time, the model receives a window of BLOCK_SIZE tokens
    and must predict each next token in that window.

    This Dataset class constructs these input/target pairs efficiently
    from the pre-tokenized `.bin` files produced by `prepare_dataset.py`.
    """

    def __init__(self, split="train"):
        """
        Load either the training or validation token dataset.

        Parameters
        ----------
        split : str
            "train" or "val".
        """

        # Pick the correct binary file
        if split == "train":
            bin_path = TRAIN_BIN
        elif split == "val":
            bin_path = VAL_BIN
        else:
            raise ValueError("Split must be 'train' or 'val'")

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Dataset not found: {bin_path}")

        print(f"Loading {split} dataset from {bin_path}")

        # Load the stored uint16 array of token IDs from the .bin file
        data = np.fromfile(bin_path, dtype=np.uint16)

        # Convert to a PyTorch tensor of type long (required by nn.Embedding)
        data = torch.tensor(data, dtype=torch.long)
        self.data = data

        # Each sample is a sliding window of size BLOCK_SIZE + 1
        self.length = len(self.data) - BLOCK_SIZE

        print(f"{split} dataset loaded: {len(self.data):,} tokens")

    def __len__(self):
        """
        Return the number of training examples available.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Return one training example for next-token prediction.

        For an index i, extract:

            chunk = tokens[i : i + BLOCK_SIZE + 1]

        Then split it into:
            x = chunk[:-1] → input sequence
            y = chunk[1:]  → target sequence (shifted by 1 token)

        The model will learn to predict y[k] from x[k].
        """

        # Slice out a window of BLOCK_SIZE+1 tokens
        chunk = self.data[idx : idx + BLOCK_SIZE + 1]

        # Input sequence (first BLOCK_SIZE tokens)
        x = chunk[:-1]

        # Target sequence (same length, but shifted by one)
        y = chunk[1:]

        return x, y


def create_dataloader(split="train"):
    """
    Create a PyTorch DataLoader for batching GPTDataset samples.

    The DataLoader:
      - shuffles only the training split
      - uses pin_memory when training on GPU for faster host - device transfer
      - drops last batch so shapes remain consistent

    Returns
    -------
    torch.utils.data.DataLoader
    """

    dataset = GPTDataset(split)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True if split == "train" else False,

        # Windows-safe (avoid multiprocessing issues)
        num_workers=0,          
        
        # Speed up transfer to GPU
        pin_memory=(device == "cuda"), 

        # Ensure each batch has consistent shape 
        drop_last=True          
    )

    return loader
