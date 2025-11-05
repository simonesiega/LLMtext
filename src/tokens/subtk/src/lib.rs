use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

/// Entry point.
#[pymodule]
fn subtk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SubwordTokenizer>()?;
    Ok(())
}

/// Subword tokenizer class.
///
/// Fields:
/// - `vocab`: the set of all learned tokens/subwords.
/// - `merges`: the ordered list of merged symbol pairs during training.
/// - `special_tokens`: tokens like <unk>, <pad>, <s>, </s>.
#[pyclass]
pub struct SubwordTokenizer {
    /// Vocabulary of learned subword tokens (exposed to Python)
    #[pyo3(get)]
    pub vocab: HashSet<String>,

    /// Ordered list of merge operations applied during training (exposed to Python)
    #[pyo3(get)]
    pub merges: Vec<(String, String)>,

    /// Special tokens reserved in the vocabulary (internal use only)
    special_tokens: Vec<String>,
}

impl SubwordTokenizer {

    /// Preprocess the input text for tokenization.
    ///
    /// # Arguments
    /// * `text` - The raw input string.
    ///
    /// # Returns
    /// A new `String` where:
    /// - All characters are lowercased.
    /// - Spaces (and any whitespace) are replaced with the visible marker '▁'.
    /// - A leading '▁' is added at the start of the string.
    fn preprocess(&self, text: &str) -> String {
        // Pre-allocate string
        let mut result = String::with_capacity(text.len() + 1);

        // Add leading marker for word boundary
        result.push('▁');

        for c in text.chars() {
            // Replace any whitespace with the '▁' marker
            if c.is_whitespace() {result.push('▁');} 
            // Lowercase non-whitespace characters
            else {result.push(c.to_ascii_lowercase());}
        }

        result
    }

    /// Count frequencies of adjacent symbol pairs in the corpus.
    ///
    /// # Arguments
    /// * `corpus` - A HashMap mapping a "word" (symbols separated by spaces) to its frequency.
    ///
    /// # Returns
    /// A HashMap where each key is a tuple `(symbol_i, symbol_j)` representing a pair
    /// of adjacent symbols, and the value is the total frequency of that pair in the corpus.
    ///
    /// Uses Rayon for parallel computation of pairs across the corpus for speedup.
    fn get_stats(&self, corpus: &HashMap<String, usize>) -> HashMap<(String, String), usize> {
        // Iterating the corpus in parallel (Rayon)
        corpus.par_iter() 
            .map(|(word, &freq)| {
                // Split the word into symbols (space-separated)
                let symbols: Vec<&str> = word.split_whitespace().collect();
                let mut local_pairs = HashMap::new();

                for i in 0..symbols.len() - 1 {
                    // Increment the count of the pair by the frequency of the word
                    *local_pairs.entry((symbols[i].to_string(), symbols[i + 1].to_string())).or_insert(0) += freq;
                }
                local_pairs
            })
            // Reduce all local maps into a single global map
            .reduce(HashMap::new, |mut acc, map| {
                // Merge counts from each local map
                for (k, v) in map {*acc.entry(k).or_insert(0) += v;}
                acc
            })
    }

    /// Merge the most frequent pair of symbols in the corpus.
    ///
    /// # Arguments
    /// * `pair` - A tuple `(String, String)` representing the symbol pair to merge.
    /// * `corpus` - A HashMap mapping "words" (symbols separated by spaces) to their frequencies.
    ///
    /// # Returns
    /// A new HashMap where every occurrence of `pair` in each word has been replaced
    /// with the merged token (concatenation of the pair).
    fn merge_pair(&self, pair: &(String, String), corpus: &HashMap<String, usize>) -> HashMap<String, usize> {
        // Pattern for replacement
        let bigram = format!("{} {}", pair.0, pair.1);
        let replacement = format!("{}{}", pair.0, pair.1);

        // Process all words in parallel
        corpus.par_iter()
            .map(|(word, &freq)| {
                // Replace occurrences of the target bigram with the merged token
                let new_word = word.replace(&bigram, &replacement);
                (new_word, freq)
            })
            // Collect results into a new HashMap
            .collect()
    }
}

#[pymethods]
impl SubwordTokenizer {    

    /// Create a new instance of `SubwordTokenizer`.
    ///
    /// Initializes:
    /// - `vocab` as an empty `HashSet<String>` to store learned tokens.
    /// - `merges` as an empty `Vec<(String, String)>` to store the BPE merge operations.
    /// - `special_tokens` with common special tokens used in NLP: `<unk>`, `<pad>`, `<s>`, `</s>`.
    ///
    /// # Returns
    /// An initialized `SubwordTokenizer` instance.
    #[new]
    pub fn new() -> Self {
        Self {
            // Vocabulary of learned subword tokens
            vocab: HashSet::new(),

            // Ordered list of merge operations applied during training
            merges: Vec::new(),

            // Special tokens
            special_tokens: vec![
                "<unk>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ],
        }
    }

    /// Train the tokenizer using a list of input texts.
    ///
    /// Implements the Byte Pair Encoding (BPE) algorithm:
    /// 1. Build an initial corpus of symbol sequences from the texts.
    /// 2. Iteratively merge the most frequent adjacent symbol pairs.
    /// 3. Construct the final vocabulary including special tokens.
    ///
    /// # Arguments
    /// * `texts` - Vector of strings containing the training corpus.
    /// * `vocab_size` - Number of merge operations (defines the size of the learned vocabulary).
    pub fn train(&mut self, texts: Vec<String>, vocab_size: usize) {
        // Build the corpus in parallel 
        // Each word is split into characters and stored with its frequency
        let corpus: HashMap<String, usize> = texts.par_iter()
            .map(|text| {
                let mut local = HashMap::new();
                // Preprocess the text (lowercase + '▁' for spaces)
                let t = self.preprocess(text);

                // Split into words by the special '▁' symbol
                for word in t.split('▁').filter(|w| !w.is_empty()) {

                    // Split each word into characters with spaces in between
                    // Example: "lower" -> "▁ l o w e r"
                    let symbols: String = "▁ ".to_string()
                        + &word.chars().map(|c| c.to_string() + " ").collect::<String>();

                    // Increment local frequency counter
                    *local.entry(symbols.trim().to_string()).or_insert(0) += 1;
                }
                local
            })
            // Merge all local HashMaps from parallel threads into a single corpus
            .reduce(HashMap::new, |mut acc, map| {
                // Merge counts from each local map
                for (k, v) in map {*acc.entry(k).or_insert(0) += v;}
                acc
            });

        // Make mutable for merging
        let mut corpus = corpus; 

        // Iteratively merge the most frequent symbol pairs
        for _ in 0..vocab_size {
            // Count pair frequencies
            let pairs = self.get_stats(&corpus); 
            
            // Stop if no more mergeable pairs
            if pairs.is_empty() {break; }

            // Select the most frequent pair
            let best_pair = pairs.iter().max_by_key(|entry| entry.1).unwrap().0.clone();

            // Merge it across the corpus
            corpus = self.merge_pair(&best_pair, &corpus);

            // Record the merge operation in order
            self.merges.push(best_pair);
        }

        // Build the final vocabulary 
        self.vocab.clear();
        for word in corpus.keys() {
            for token in word.split_whitespace() {
                self.vocab.insert(token.to_string());
            }
        }

        // Add special tokens to vocabulary
        self.vocab.extend(self.special_tokens.clone());
    }

    /// Encode a text string into a list of subword tokens using the learned merges.
    ///
    /// # Arguments
    /// * `text` - The input string to tokenize.
    ///
    /// # Returns
    /// * `Vec<String>` - Vector of subword tokens representing the input text.
    pub fn encode(&self, text: &str) -> Vec<String> {
        // Step 1: Preprocess input text (normalize)
        let t = self.preprocess(text);

        // Step 2: Convert text to a vector of individual character strings
        let mut symbols: Vec<String> = t.chars().map(|c| c.to_string()).collect();

        // Step 3: Apply all learned merges in order
        for (f, s) in &self.merges {
            let mut i = 0;

            // Prepare a new vector to hold symbols after this merge
            let mut new_symbols = Vec::with_capacity(symbols.len());

            while i < symbols.len() {
                // If consecutive symbols match the current merge pair, merge them
                if i + 1 < symbols.len() && symbols[i] == *f && symbols[i + 1] == *s {
                    // Merge pair into one token
                    new_symbols.push(format!("{}{}", f, s));

                    // Skip the next symbol since it was merged
                    i += 2;
                } 
                else {
                    new_symbols.push(symbols[i].clone()); 
                    i += 1;
                }
            }

            // Update symbols with the merged version for the next iteration
            symbols = new_symbols;
        }

        // Return the final list of subword tokens
        symbols
    }

    /// Decode a list of subword tokens back into a readable text string.
    ///
    /// # Arguments
    /// * `tokens` - A vector of subword tokens to decode.
    ///
    /// # Returns
    /// * `String` - The reconstructed text.
    pub fn decode(&self, tokens: Vec<String>) -> String {
        // Step 1: Concatenate all tokens into a single string
        // Step 2: Replace '▁' with space to reconstruct word boundaries
        // Step 3: Trim leading/trailing whitespace and return
        tokens.join("").replace("▁", " ").trim().to_string()
    }
}