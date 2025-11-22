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
    
    @property
    def token_to_id(self):
        """
        Returns dict: token -> id (int)

        Rust side: SubwordTokenizer.token_to_id.
        """
        return self._rust_tok.token_to_id
    
    @property
    def id_to_token(self):
        """
        Returns List[str] where index is token id.

        Rust side: SubwordTokenizer.id_to_token getter.
        """
        return self._rust_tok.id_to_token
    
    @property
    def special_tokens(self):
        """Return special tokens configured inside Rust."""
        return self._rust_tok.special_tokens
    
    # ---------------------------

    def train(self, texts, max_merges=100, min_freq_token=3):
        """
        Train the tokenizer on a list of input texts.

        Parameters
        ----------
        texts : list of str
            Input sentences or documents to train on.
        max_merges : int
            Maximum number of merges to perform.

        Rust side: calls `SubwordTokenizer::train()`
        """
        # Ensure all inputs are strings
        texts = [str(t) for t in texts]
        self._rust_tok.train(texts, max_merges, min_freq_token)
    
    # ---------------------------

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

    def encode_ids(self, text):
        """
        Encode a text into a list of token IDs.

        Rust: SubwordTokenizer.encode_ids
        """
        return self._rust_tok.encode_ids(str(text))

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

    def decode_from_ids(self, ids):
        """
        Decode a list of token IDs into a text.

        Rust: SubwordTokenizer.decode_from_ids
        """
        ids = [int(i) for i in ids]
        return self._rust_tok.decode_from_ids(ids)
    
    # ---------------------------

    def save(self, path):
        """
        Save tokenizer state (vocab, merges, special tokens) to JSON file.

        Rust: SubwordTokenizer.save_to_file
        """
        return self._rust_tok.save_to_file(str(path))

    @staticmethod
    def load(path):
        """
        Load tokenizer from a JSON file and return a new wrapper instance.

        Rust: SubwordTokenizer.load_from_file
        """
        wrapper = SubwordTokenizerRust.__new__(SubwordTokenizerRust)
        wrapper._rust_tok = SubwordTokenizer.load_from_file(str(path))
        return wrapper