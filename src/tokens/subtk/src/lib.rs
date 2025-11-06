use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use std::sync::Arc;

/// Entry point for the Python module
#[pymodule]
fn subtk(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the SubwordTokenizer class so it is accessible from Python
    m.add_class::<SubwordTokenizer>()?;
    Ok(())
}

/// Subword tokenizer class
#[pyclass]
pub struct SubwordTokenizer {
    // Internal vocabulary of learned subword tokens.
    vocab_internal: HashSet<Arc<str>>,

    // Ordered list of merge operations applied during training.
    // Each tuple represents a pair of symbols merged into one token.
    merges_internal: Vec<(Arc<str>, Arc<str>)>,

    // Special tokens reserved in the vocabulary (e.g., <unk>, <pad>, <s>, </s>)
    special_tokens: Vec<String>,
}

impl SubwordTokenizer {
    /// Preprocess the input text for tokenization.
    ///
    /// This method normalizes the text by:
    /// 1. Converting all characters to lowercase.
    /// 2. Replacing any whitespace with the visible marker '▁'.
    /// 3. Adding a leading '▁' at the beginning of the string to indicate word boundary.
    ///
    /// # Arguments
    /// * `text` - A string slice containing the raw input text.
    ///
    /// # Returns
    /// A new `String` with preprocessing applied, ready for tokenization.
    fn preprocess(&self, text: &str) -> String {
        // Pre-allocate string 
        let mut result = String::with_capacity(text.len() + 1);

        result.push('▁');

        // Iterate over each character
        for c in text.chars() {
            // Replace whitespace with '▁' marker
            if c.is_whitespace() { result.push('▁'); }
            // Convert non-whitespace characters to lowercase
            else { result.push(c.to_ascii_lowercase()); }
        }
        result
    }

    /// Count the frequency of adjacent symbol pairs in the corpus.
    ///
    /// # Arguments
    /// * `corpus` - A HashMap where keys are words represented as space-separated symbols (Arc<str>),
    ///              and values are their frequency counts.
    ///
    /// # Returns
    /// A HashMap where keys are pairs of symbols `(Arc<str>, Arc<str>)` and values are
    /// the total frequency counts of each pair in the corpus.
    fn get_stats(&self, corpus: &HashMap<Arc<str>, usize>)
        -> HashMap<(Arc<str>, Arc<str>), usize>
    {
        corpus.par_iter()
            .map(|(word, &freq)| {
                // Split the word into individual symbols
                let symbols: Vec<_> = word.split_whitespace().map(Arc::<str>::from).collect();

                // Preallocate a local map to count pairs
                let mut local = HashMap::with_capacity(symbols.len().saturating_sub(1));

                // Iterate over windows of 2 symbols to count pairs
                for pair in symbols.windows(2) {
                    *local.entry((pair[0].clone(), pair[1].clone())).or_insert(0) += freq;
                }

                local
            })
            // Reduce all local maps into a single global map
            .reduce(HashMap::new, |mut acc, map| {
                for (k, v) in map {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            })
    }

    /// Merge a given pair of symbols throughout the corpus.
    ///
    /// This method applies a single Byte Pair Encoding (BPE) merge operation:
    /// every occurrence of the specified symbol pair `(first, second)` in each word
    /// is replaced by their concatenation.
    ///
    /// # Arguments
    /// * `pair` - A tuple `(Arc<str>, Arc<str>)` representing the symbol pair to merge.
    /// * `corpus` - A HashMap where keys are words represented as space-separated symbols
    ///              and values are their frequency counts.
    ///
    /// # Returns
    /// A new HashMap with the same structure as `corpus`, but with the specified
    /// pair merged wherever it occurs.
    fn merge_pair(
        &self,
        pair: &(Arc<str>, Arc<str>),
        corpus: &HashMap<Arc<str>, usize>,
    ) -> HashMap<Arc<str>, usize>
    {
        let (first, second) = (&pair.0, &pair.1);

        // Create the merged token by concatenating the two symbols
        let merged = format!("{}{}", first, second);

        corpus.par_iter()
            .map(|(word, &freq)| {
                // Split the word into its current symbols
                let symbols: Vec<&str> = word.split_whitespace().collect();
                let mut out = Vec::with_capacity(symbols.len());
                let mut i = 0;

                // Iterate through the symbols and merge the target pair
                while i < symbols.len() {
                    if i + 1 < symbols.len()
                        && symbols[i] == first.as_ref()
                        && symbols[i + 1] == second.as_ref()
                    {
                        // Merge the pair into one token
                        out.push(merged.clone());
                        // Skip the next symbol since it was merged
                        i += 2; 
                    } 
                    else {
                        // Keep the current symbol unchanged
                        out.push(symbols[i].to_string());
                        i += 1;
                    }
                }

                // Join symbols back into a space-separated word and preserve frequency
                (Arc::from(out.join(" ")), freq)
            })
            // Collect all transformed words into a new corpus
            .collect()
    }
}

#[pymethods]
impl SubwordTokenizer {
    /// Create a new instance of `SubwordTokenizer`.
    ///
    /// Initializes internal data structures for the tokenizer:
    /// - `vocab_internal` stores learned subword tokens efficiently as `Arc<str>`.
    /// - `merges_internal` stores the ordered list of BPE merge operations.
    /// - `special_tokens` contains common reserved tokens for NLP: `<unk>`, `<pad>`, `<s>`, `</s>`.
    ///
    /// # Returns
    /// A fully initialized `SubwordTokenizer`.
    #[new]
    pub fn new() -> Self {
        Self {
            // Internal vocabulary: empty initially
            vocab_internal: HashSet::new(),

            // Internal list of merges applied during training
            merges_internal: Vec::new(),

            // Reserved special tokens
            special_tokens: vec![
                "<unk>".to_string(), // Unknown token
                "<pad>".to_string(), // Padding token
                "<s>".to_string(),   // Start-of-sequence token
                "</s>".to_string(),  // End-of-sequence token
            ],
        }
    }

    /// Train the tokenizer on a given set of texts using Byte Pair Encoding (BPE).
    ///
    /// # Arguments
    /// * `texts` - Vector of input strings to build the vocabulary from.
    /// * `vocab_size` - Maximum number of merge operations to perform (controls vocabulary size).
    ///
    /// # Description
    /// 1. Build an initial corpus of symbol sequences where each word is split into characters.
    /// 2. Count the frequency of each sequence across all texts in parallel.
    /// 3. Iteratively merge the most frequent adjacent symbol pairs up to `vocab_size` times.
    /// 4. Construct the final vocabulary including the learned subwords and special tokens.
    pub fn train(&mut self, texts: Vec<String>, vocab_size: usize) {
        // Step 1: Build corpus in parallel
        // Each word is split into characters with a leading '▁' marker for word boundaries
        let corpus: HashMap<Arc<str>, usize> = texts.par_iter()
            .map(|text| {
                let mut local = HashMap::new();
                let t = self.preprocess(text);

                // Split text into words based on '▁' and ignore empty segments
                for word in t.split('▁').filter(|w| !w.is_empty()) {
                    // Preallocate a string for the symbol sequence
                    let mut symbols = String::with_capacity(word.len() * 2 + 2);
                    // Leading marker
                    symbols.push_str("▁ "); 

                    for c in word.chars() {
                        symbols.push(c);    
                        symbols.push(' ');  
                    }

                    // Count frequency of the symbol sequence
                    *local.entry(Arc::from(symbols.trim())).or_insert(0) += 1;
                }
                local
            })
            // Merge all local hashmaps into a single corpus
            .reduce(HashMap::new, |mut acc, map| {
                for (k, v) in map { *acc.entry(k).or_insert(0) += v; }
                acc
            });
        
        // Make mutable for iterative merges
        let mut corpus = corpus; 

        // Step 2: Iteratively merge the most frequent pairs
        for _ in 0..vocab_size {
            // Compute frequency of adjacent symbol pairs
            let pairs = self.get_stats(&corpus);

            // Stop if no more pairs to merge
            if pairs.is_empty() { break; }

            // Select the most frequent pair
            let best_pair = pairs.iter().max_by_key(|entry| entry.1).unwrap().0.clone();

            // Merge the pair across the corpus
            corpus = self.merge_pair(&best_pair, &corpus);

            // Record the merge operation in order
            self.merges_internal.push(best_pair);
        }

        // Step 3: Build the final vocabulary from the corpus
        self.vocab_internal.clear();
        for word in corpus.keys() {
            for token in word.split_whitespace() {
                self.vocab_internal.insert(Arc::from(token));
            }
        }

        // Step 4: Add special tokens to the vocabulary
        self.vocab_internal.extend(self.special_tokens.iter().map(|s| Arc::from(s.as_str())));
    }

    /// Encode a text string into a list of subword tokens using the learned merges.
    ///
    /// # Arguments
    /// * `text` - The input string to tokenize.
    ///
    /// # Returns
    /// * `Vec<String>` - A vector of subword tokens representing the input text.
    ///
    /// # Description
    /// 1. Preprocess the input text (lowercase + leading '▁' + replace spaces with '▁').
    /// 2. Convert the text into a vector of individual characters as initial symbols.
    /// 3. Apply all learned merge operations in order to combine symbols into subwords.
    pub fn encode(&self, text: &str) -> Vec<String> {
        // Step 1: Preprocess input text
        let t = self.preprocess(text);

        // Step 2: Convert text into a vector of characters (symbols)
        let mut symbols: Vec<String> = t.chars().map(|c| c.to_string()).collect();

        // Step 3: Iteratively apply all BPE merges
        for (f, s) in &self.merges_internal {
            let mut i = 0;
            // Prepare vector for merged symbols
            let mut new_symbols = Vec::with_capacity(symbols.len()); 

            while i < symbols.len() {
                // Check if the current symbol and the next match the merge pair
                if i + 1 < symbols.len()
                    && symbols[i] == f.as_ref()
                    && symbols[i + 1] == s.as_ref()
                {
                    // Merge the pair into a single token
                    new_symbols.push(format!("{}{}", f, s));
                    // Skip the next symbol since it was merged
                    i += 2;
                } 
                else {
                    // Keep the current symbol 
                    new_symbols.push(symbols[i].clone());
                    i += 1;
                }
            }

            // Update symbols for the next merge iteration
            symbols = new_symbols;
        }

        // Return the final list
        symbols
    }

    /// Decode a list of subword tokens back into a readable string.
    ///
    /// # Arguments
    /// * `tokens` - A vector of subword tokens to decode.
    ///
    /// # Returns
    /// * `String` - The reconstructed text.
    pub fn decode(&self, tokens: Vec<String>) -> String {
        tokens.join("").replace("▁", " ").trim().to_string()
    }

    /// Getter for the tokenizer's vocabulary exposed to Python.
    ///
    /// # Returns
    /// * `Vec<String>` - The set of learned subword tokens as a vector of strings.
    ///
    /// # Description
    /// Converts the internal `HashSet<Arc<str>>` into a `Vec<String>`
    /// so that Python can access it in a natural format.
    #[getter]
    pub fn vocab(&self) -> Vec<String> {
        self.vocab_internal.iter().map(|s| s.to_string()).collect()
    }

    /// Getter for the list of merge operations exposed to Python.
    ///
    /// # Returns
    /// * `Vec<(String, String)>` - Vector of all BPE merge operations as string pairs.
    ///
    /// # Description
    /// Converts the internal `Vec<(Arc<str>, Arc<str>)>` into a vector of `(String, String)`
    /// so Python can access the merge sequence in a readable format.
    #[getter]
    pub fn merges(&self) -> Vec<(String, String)> {
        self.merges_internal
            .iter()
            .map(|(f, s)| (f.to_string(), s.to_string()))
            .collect()
    }
}
