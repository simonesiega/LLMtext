import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Dataset per LLM su caratteri (Char-Level Language Model).

    Ogni esempio consiste in:
    - x: sequenza di token di lunghezza block_size
    - y: stessa sequenza shiftata di 1 token (target successivo)
    
    Il modello impara a prevedere il prossimo carattere dato il contesto.
    """

    def __init__(self, text: str, block_size: int = 128):
        """
        Parameters
        ----------
        text : str
            Il testo da cui costruire il dataset.
        block_size : int
            Numero di caratteri in ogni sequenza di input.
        """

        # Lista dei caratteri unici presenti nel dataset
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # stoi = "string to index": converte ogni carattere in un ID numerico
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        
        # itos = "index to string": mappa inversa per decodificare token in caratteri
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Lunghezza delle sequenze di input
        self.block_size = block_size

        # Ogni carattere del testo diventa il suo ID numerico
        # Tipo torch.long perché rappresentano interi
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        """
        Numero di sequenze presenti nel dataset.

        Ogni sequenza é composta da block_size token
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        """
        Restituisce una coppia (x, y).

        x : torch.Tensor
            Sequenza di token di lunghezza block_size.
        y : torch.Tensor
            Sequenza target di lunghezza block_size,
            shiftata di 1 posizione a destra.

        Questo permette al modello di prevedere il prossimo token
        dato il contesto precedente.
        """
        # x = sequenza di input
        x = self.data[idx:idx + self.block_size]
        
        # y = sequenza target, shiftata di 1
        y = self.data[idx + 1:idx + self.block_size + 1]
        
        return x, y