import re
from collections import Counter

class SubwordTokenizer:

    def __init__(self):
        self.vocab = set()      
        self.merges = []        
        self.special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]


    def _preprocess(self, text: str):
        """
        Normalize text: 
        - Lowercase
        - Replace spaces with a visible marker (▁)
        """
        text = text.lower().strip()
        return "▁" + text.replace(" ", "▁")


    def _get_stats(self, corpus):
        """
        Count frequency of symbol pairs in the corpus.
        Each word is a list of symbols (space-separated string).

        Returns
        -------
        pairs : collections.Counter
            Dictionary-like object where keys are tuples (symbol_i, symbol_j)
            and values are the number of occurrences of that pair in the corpus.
        """
        pairs = Counter()
        
        # Each word and its frequency in the corpus.
        for word, freq in corpus.items():

            symbols = word.split()

            # Loop through all adjacent symbol pairs in the word.
            # Example: "l o w e r" → ("l", "o"), ("o", "w"), ("w", "e"), ("e", "r")
            for i in range(len(symbols) - 1):
                # Increment the count of this pair by the frequency of the word.
                pairs[(symbols[i], symbols[i + 1])] += freq

        # Return all counted pairs as a Counter object {('l','o'): 8, ...})
        return pairs


    def _merge_pair(self, pair, corpus):
        """
        Merge the most frequent pair of symbols into one token.

        Returns
        -------
        new_corpus : dict
            A new version of the corpus where every occurrence of the chosen pair
            has been replaced by a merged symbol (e.g. "l o" → "lo").
        """
        new_corpus = {}

        # Join the pair into a single string with a space between, e.g. ('l', 'o') → 'l o'
        bigram = ' '.join(pair)

        replacement = ''.join(pair)

        # Regex pattern to match the bigram as a whole unit between spaces
        # (?<!\S) ensures the pair is preceded only by whitespace or start of line
        # (?!\S) ensures it's followed only by whitespace or end of line
        # avoids merging parts of longer tokens accidentally
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')

        # Each word and its frequency in the corpus
        for word, freq in corpus.items():
            # Replace all occurrences of the target bigram with the merged token
            new_word = pattern.sub(replacement, word)

            new_corpus[new_word] = freq

        # Return the new corpus with the merged tokens
        return new_corpus


    def train(self, texts, vocab_size=100):
        """
        Train the Byte Pair Encoding (BPE) tokenizer on a list of input texts.

        Implements the BPE training loop:
        1. Initialize a character-level vocabulary.
        2. Repeatedly find and merge the most frequent symbol pairs.
        3. Build the final vocabulary containing learned subword units.
        """
        # Step 1 — Preprocess and initialize character-level vocabulary
        corpus = Counter()
        for text in texts:
            # Apply preprocessing
            text = self._preprocess(text)

            for word in text.split("▁"):
                # skip empty tokens
                if word: 

                    # Split each word into characters and rejoin with spaces
                    # e.g. "lower" → "▁ l o w e r"
                    word = "▁ " + " ".join(list(word))

                    # Add this word (symbol sequence) to the corpus with its frequency
                    corpus[word.strip()] += 1

        # Step 2 — Iteratively merge pairs
        for i in range(vocab_size):
            # Count all adjacent symbol pairs and their frequencies
            pairs = self._get_stats(corpus)

            # Stop if there are no more mergeable pairs (e.g. all unique)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            # Merge that pair across the corpus (e.g. ('l', 'o') → "lo")
            corpus = self._merge_pair(best_pair, corpus)

            self.merges.append(best_pair)

        # Step 3 — Build vocabulary
        self.vocab = set()
        for word in corpus:

            # Split each word into its tokens and add to the vocabulary
            for token in word.split():
                self.vocab.add(token)

        self.vocab.update(self.special_tokens)


    def encode(self, text):
        """
        Encode text into a list of subword tokens.

        Returns
        -------
        symbols : list[str]
            A list of subword tokens representing the input text.
        """
        # Apply preprocessing
        text = self._preprocess(text)

        # Split text into individual characters
        symbols = list(text)

        # Apply all merges learned during training
        # self.merges contains the merge order:
        # [(f, s), (f, s), ...] --> [('l', 'o'), ('lo','w'), ...]
        for f, s in self.merges:
            merged = f + s
            i = 0

            # Loop through symbols to find consecutive pairs (f, s)
            while i < len(symbols) - 1:
                # Match f,s found
                if symbols[i] == f and symbols[i + 1] == s:
                    # Merge the pair in place: replace symbols[i] and symbols[i+1] with [merged_token]
                    symbols[i:i+2] = [merged]
                
                else:
                    # Move to the next symbol if no merge occurs
                    i += 1
        return symbols

    def decode(self, tokens):
        """
        Decode list of tokens back to a text string.

        Returns
        -------
        text : str
            The reconstructed text string.
        """
        # Join all tokens into a single string and replace ▁ with spaces
        return "".join(tokens).replace("▁", " ").strip()
