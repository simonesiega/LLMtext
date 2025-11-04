import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Character-level Language Modeling Dataset.

    Each training example consists of:
    - x: a sequence of `block_size` tokens (input)
    - y: the same sequence shifted by one token (target)

    The model learns to predict the next character given the previous context.
    """

    def __init__(self, text: str, block_size: int = 128):
        """
        Parameters
        ----------
        text : str
            The text corpus used to build the dataset.
        block_size : int
            Number of characters in each input sequence.
        """

        # Unique characters appearing in the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # stoi = "string to index": maps each character to a unique integer ID
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        
        # itos = "index to string": inverse mapping for decoding
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Sequence length for each input sample
        self.block_size = block_size

        # Convert the entire text into a sequence of integer token IDs
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len_sliding__(self) -> int:
        """
        Compute the number of training samples using a sliding window.

        Each input sequence is a window of size `block_size`
        that moves by one character at a time over the text.

        Returns
        -------
        int
            Total number of overlapping training samples.
        """
        return len(self.data) - self.block_size

    def __len_chunked__(self) -> int:
        """
        Compute the number of training samples using non-overlapping chunks.

        The text is divided into consecutive, independent segments
        of size `block_size`.

        Returns
        -------
        int
            Total number of disjoint chunks in the dataset.
        """
        return len(self.data) // self.block_size

    def __getitem__(self, idx: int):
        """
        Retrieve a pair (x, y) for training.

        Parameters
        ----------
        idx : int
            Starting index of the sequence.

        Returns
        -------
        (x, y) : Tuple[torch.Tensor, torch.Tensor]
            - x: sequence of length `block_size`
            - y: same sequence shifted by one position

        The target y represents the next character for each token in x.
        """
        # Input sequence
        x = self.data[idx:idx + self.block_size]
        
        # Target sequence (shifted by one position)
        y = self.data[idx + 1:idx + self.block_size + 1]
        
        return x, y
