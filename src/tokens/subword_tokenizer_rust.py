from subtk import SubwordTokenizer

class SubwordTokenizerRust:
    """
    Python wrapper for the Rust-based SubwordTokenizer.

    This class exposes a Python interface and delegates
    all heavy lifting to the Rust implementation.
    """

    def __init__(self):
        # Initialize the Rust tokenizer instance
        # Calls Rust `SubwordTokenizer::new()`
        self._rust_tok = SubwordTokenizer()

    @property
    def vocab(self):
        """
        Return the vocabulary of learned subword tokens.

        Rust side: accesses `SubwordTokenizer.vocab` (HashSet<String>)
        """
        return self._rust_tok.vocab

    @property
    def merges(self):
        """
        Return the list of merge operations performed during training.

        Rust side: accesses `SubwordTokenizer.merges` (Vec<(String, String)>)
        """
        return self._rust_tok.merges

    def train(self, texts, vocab_size=100):
        """
        Train the tokenizer on a list of input texts.

        Parameters
        ----------
        texts : list of str
            Input sentences or documents to train on.
        vocab_size : int
            Maximum number of merges to perform (controls vocabulary size).

        Rust side: calls `SubwordTokenizer::train()`
        """
        # Ensure all inputs are strings
        texts = [str(t) for t in texts]
        self._rust_tok.train(texts, vocab_size)

    def encode(self, text):
        """
        Convert a text string into a list of subword tokens.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        List[str]
            Subword tokens representing the input text.

        Rust side: calls `SubwordTokenizer::encode()`
        """
        return self._rust_tok.encode(str(text))

    def decode(self, tokens):
        """
        Reconstruct text from a list of subword tokens.

        Parameters
        ----------
        tokens : list of str
            Tokenized subwords.

        Returns
        -------
        str
            Decoded text string.

        Rust side: calls `SubwordTokenizer::decode()`
        """
        return self._rust_tok.decode(list(tokens))