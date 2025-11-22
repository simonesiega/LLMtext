use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError, PyRuntimeError};
use serde::{Serialize, Deserialize};
use std::collections::{BinaryHeap};
use std::sync::Arc;
use std::path::Path;
use std::fs;
use std::fmt::Write as FmtWrite;
use std::usize;
use smallvec::SmallVec;
use rustc_hash::FxHashMap;

/// Converts a byte array to a hexadecimal string.
///
/// # Parameters
/// - `b`: a slice of bytes (`&[u8]`) to convert.
///
/// # Returns
/// - `String` containing the hexadecimal representation of the bytes.
fn bytes_to_hex(b: &[u8]) -> String {
    // Preallocate a string with capacity double the length of the byte array,
    // each byte will become two hexadecimal characters
    let mut s = String::with_capacity(b.len() * 2);
    
    for &x in b {
        // Write the byte as a two-digit hexadecimal into the string
        write!(s, "{:02x}", x).unwrap();
    }

    // Return the resulting string
    s
}

/// Converts a hexadecimal string to a vector of bytes.
///
/// # Parameters
/// - `s`: a string slice (`&str`) containing hexadecimal characters. Must have an even length.
///
/// # Returns
/// - `Ok(Vec<u8>)` with the corresponding bytes if parsing succeeds.
/// - `Err(String)` with an error message if the string has odd length or contains invalid hex.
fn hex_to_bytes(s: &str) -> Result<Vec<u8>, String> {
    // Ensure the hex string has an even number of characters
    if s.len() % 2 != 0 { return Err("odd hex length".to_string()); }

    // Preallocate output vector with half the length of the hex string
    let mut out = Vec::with_capacity(s.len() / 2);

    for i in (0..s.len()).step_by(2) {
        // Parse each two-character chunk as a hexadecimal byte
        let byte = u8::from_str_radix(&s[i..i+2], 16)
            .map_err(|e| format!("hex parse error: {}", e))?;

        out.push(byte);
    }

    // Return the resulting byte vector
    Ok(out)
}

/// A pool of unique tokens.
///
/// Maps byte sequences to unique IDs and stores the actual bytes for each ID.
///
/// # Fields
/// - `ids`: `FxHashMap<Vec<u8>, usize>` — maps byte sequences to their unique ID.
/// - `tokens`: `Vec<Arc<[u8]>>` — maps ID to the corresponding byte sequence.
#[derive(Default)]
struct TokenPool {
    // bytes -> id
    ids: FxHashMap<Vec<u8>, usize>, 
    // id -> bytes
    tokens: Vec<Arc<[u8]>>,         
}

impl TokenPool {
    /// Creates a new, empty `TokenPool`.
    ///
    /// # Returns
    /// - `TokenPool` with empty `ids` and `tokens`.
    fn new() -> Self {
        // init 
        Self { 
            ids: FxHashMap::default(), 
            tokens: Vec::new() 
        }
    }

    /// Interns a byte sequence into the pool, returning its unique ID.
    ///
    /// If the byte sequence already exists in the pool, its existing ID is returned.
    /// Otherwise, a new ID is assigned, and the bytes are stored.
    ///
    /// # Parameters
    /// - `bytes`: `Vec<u8>` representing the token to intern.
    ///
    /// # Returns
    /// - `usize`: the unique ID of the interned token.
    fn intern(&mut self, bytes: Vec<u8>) -> usize {
        // Check if the byte sequence is already interned
        if let Some(&id) = self.ids.get(&bytes) {
            // return existing ID
            return id; 
        }

        // Assign a new ID based on current token count
        let id = self.tokens.len();

        // Store the bytes as Arc<[u8]>
        self.tokens.push(Arc::from(bytes.clone().into_boxed_slice()));

        // Map the byte sequence to the new ID
        self.ids.insert(bytes, id);

        // Return the assigned ID
        id
    }

    /// Returns a reference to the byte sequence corresponding to a given token ID.
    ///
    /// # Parameters
    /// - `id`: `usize` — the unique ID of the token.
    ///
    /// # Returns
    /// - `&[u8]` — a slice of the bytes stored for this token ID.
    fn get_bytes(&self, id: usize) -> &[u8] {
        // Access the Arc<[u8]> at index `id` and return it as a byte slice
        self.tokens[id].as_ref()
    }

    /// Returns the number of tokens currently stored in the pool.
    ///
    /// # Returns
    /// - `usize` — the number of interned tokens.
    ///
    /// # Example
    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns all tokens in the pool as hexadecimal strings.
    ///
    /// # Returns
    /// - `Vec<String>` — a vector of hex strings representing each token's byte sequence.
    fn tokens_hex(&self) -> Vec<String> {
        // Iterate over all tokens, convert each byte slice to a hex string, collect into a vector
        self.tokens.iter().map(|a| bytes_to_hex(a.as_ref())).collect()
    }
}

/// Create a unique 64-bit key from a pair of token IDs.
/// 
/// # Arguments
/// * `a` - First token ID (usize)
/// * `b` - Second token ID (usize)
///
/// # Returns
/// * `u64` - A 64-bit key where `a` occupies the high 32 bits and `b` the low 32 bits.
#[inline]
fn pair_key(a: usize, b: usize) -> u64 {
    // Shift `a` to the high 32 bits and combine with `b` in the low 32 bits
    ((a as u64) << 32) | (b as u64)
}

/// Extract the original token IDs from a 64-bit pair key.
/// 
/// # Arguments
/// * `k` - 64-bit key created by `pair_key`.
///
/// # Returns
/// * `(usize, usize)` - Tuple containing the original token IDs `(a, b)`.
#[inline]
fn unpack_pair(k: u64) -> (usize, usize) {
    // High 32 bits is `a`, low 32 bits is `b`
    (((k >> 32) as usize), (k as u32 as usize))
}

/// Struct used for serializing and deserializing a trained tokenizer.
///
/// This struct stores all the information needed to save a tokenizer
/// and later reconstruct it.
#[derive(Serialize, Deserialize)]
struct SerializableTokenizer {
    /// List of all tokens in hexadecimal string format.
    vocab: Vec<String>,

    /// List of merge operations as pairs of hexadecimal token strings.
    /// Each tuple represents a merge applied during BPE training.
    merges: Vec<(String, String)>,

    /// List of special tokens.
    special_tokens: Vec<String>,
}

/// Python module definition for exposing `SubwordTokenizer` to Python via PyO3.
///
/// This initializes the `subtk` module and registers the `SubwordTokenizer`
/// class so it can be imported and used directly from Python.
///
/// # Returns
/// * `PyResult<()>` — Indicates whether module initialization succeeded.
#[pymodule]
fn subtk(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the SubwordTokenizer class so it becomes available in Python
    m.add_class::<SubwordTokenizer>()?;
    Ok(())
}

/// A subword tokenizer implementation using an arena-based corpus representation.
///
/// This struct supports:
/// - BPE-style training
/// - Deterministic encoding/decoding
///
/// The internal design focuses on speed, locality, and minimal allocations:
/// raw bytes are deduplicated in a `TokenPool`, and the corpus is represented
/// as a flat arena of token slots with explicit linked-list navigation.
///
/// # Main Components
/// ## Token Storage
/// * `pool`: Interns raw byte sequences and assigns stable token IDs.
/// * `merges_internal`: Sequence of merges learned during training
///   as `(id_left, id_right)` pairs.
/// * `special_tokens`: User-provided tokens that bypass BPE merging.
///
/// ## Runtime Mappings
/// * `token_to_id`: token → integer ID for encoding.
/// * `id_to_token`: Reverse mapping for decoding.
///
/// ## Training Arena
/// During training, each token in the corpus becomes a node in an arena:
/// * `arena_token_ids`: Token ID stored at each arena index.
/// * `arena_alive`: Whether the node is still active (not merged away).
/// * `arena_prev`, `arena_next`: Linked-list style navigation for O(1)
///    removal/merge operations without reallocating or shifting data.
///
/// Words are stored as slices of the arena:
/// * `word_starts`: Arena index where each word begins.
/// * `word_lens`: Number of arena slots that belong to each word.
///
/// The struct is exposed to Python.
#[pyclass]
pub struct SubwordTokenizer {
    // token pool: interns raw byte sequences and assigns stable token IDs
    pool: TokenPool,

    // merges as pairs of token IDs (in the order they were learned)
    merges_internal: Vec<(usize, usize)>,

    // special tokens protected from merging
    special_tokens: Vec<String>,

    // runtime mappings for encode/decode
    token_to_id: FxHashMap<Arc<[u8]>, u32>,
    id_to_token: Vec<Arc<[u8]>>,

    // token id at each arena index
    arena_token_ids: Vec<usize>,
    // whether the arena node is still active
    arena_alive: Vec<bool>,
    // linked-list navigation (usize::MAX means no neighbor)
    arena_prev: Vec<usize>,
    arena_next: Vec<usize>,

    // per-word arena ranges
    word_starts: Vec<usize>,
    word_lens: Vec<usize>,
}

#[allow(dead_code)]
impl SubwordTokenizer {
    /// Creates a new `SubwordTokenizer` with a predefined set of special tokens
    /// and an initialized byte-level vocabulary.
    ///
    /// This constructor:
    /// - Registers the standard special tokens:
    ///     * `<unk>` – unknown token  
    ///     * `<pad>` – padding token  
    ///     * `<s>`   – start of sequence  
    ///     * `</s>`  – end of sequence  
    /// - Interns all special tokens into the `TokenPool` first, ensuring they
    ///   receive the lowest token IDs and remain stable across training.
    /// - Pre-interns all single-byte tokens (`0..=255`)
    ///
    /// # Returns
    /// An initialized tokenizer ready to be trained or used for raw byte segmentation.
    fn new_with_specials() -> Self {
        let mut t = SubwordTokenizer {
            pool: TokenPool::new(),
            merges_internal: Vec::new(),
            special_tokens: vec![
                "<unk>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ],
            token_to_id: FxHashMap::default(),
            id_to_token: Vec::new(),
            arena_token_ids: Vec::new(),
            arena_alive: Vec::new(),
            arena_prev: Vec::new(),
            arena_next: Vec::new(),
            word_starts: Vec::new(),
            word_lens: Vec::new(),
        };

        // intern special tokens first
        for s in &t.special_tokens {
            t.pool.intern(s.as_bytes().to_vec());
        }
        // pre-intern single bytes 0..=255 
        for b in 0u8..=255u8 {
            t.pool.intern(vec![b]);
        }
        
        // Return the instance
        t
    }

    /// Preprocesses a string into a normalized byte sequence suitable for training.
    ///
    /// - Converts all characters to lowercase
    /// - Replaces any whitespace with a single ASCII space (`0x20`)
    /// - Encodes all characters into UTF-8 bytes
    /// - Prepends one space at the beginning 
    ///
    /// The result is a byte vector where words are separated by single spaces
    /// and all characters are UTF-8 encoded.
    ///
    /// # Arguments
    /// * `text` – The input UTF-8 string to preprocess
    ///
    /// # Returns
    /// A `Vec<u8>` containing the normalized UTF-8 bytes.
    fn preprocess_bytes(text: &str) -> Vec<u8> {
        // Convert text to lowercase first (normalization)
        let lower = text.to_lowercase();

        // Reserve enough capacity 
        let mut out = Vec::with_capacity(lower.len() + 1);
        out.push(b' ');


        for ch in lower.chars() {
            if ch.is_whitespace() {
                // Normalize any whitespace down to a single ASCII space
                out.push(b' ');
            } 
            else {
                // Encode character as UTF-8 bytes
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                out.extend_from_slice(s.as_bytes());
            }
        }
        
        // Return the normalized byte sequence
        out
    }

    /// Rebuilds the runtime token → ID lookup tables from the internal `TokenPool`.
    ///
    /// This is called after training finishes (or after loading from disk) to
    /// construct efficient mappings for encoding/decoding at inference time.
    ///
    /// The mapping is stable because token IDs in the pool never change once
    /// interned.
    ///
    /// # Effects
    /// Updates:
    /// - `self.id_to_token`  
    /// - `self.token_to_id`
    fn build_mappings_from_pool(&mut self) {
        // Copy all tokens from the pool into a flat indexable list (id → token bytes)
        self.id_to_token = self.pool.tokens.clone();

        // Pre-allocate the FxHashMap for ID lookup (token bytes → id)
        let mut map = FxHashMap::with_capacity_and_hasher(
            self.id_to_token.len(),
            Default::default(),
        );

        // Fill reverse mapping
        for (i, tok) in self.id_to_token.iter().enumerate() {
            // Insert cloned Arc<[u8]> as key, id as u32
            map.insert(tok.clone(), i as u32);
        }

        // Replace old map
        self.token_to_id = map;
    }

    /// Returns the previous alive arena index for a given slot.
    ///
    /// The arena uses `usize::MAX` as a sentinel to flag “no previous element”.
    /// This helper converts that sentinel into an idiomatic `Option<usize>`
    /// so higher-level code can work safely without handling raw sentinels.
    ///
    /// # Arguments
    /// * `idx` – The current arena index
    ///
    /// # Returns
    /// `Some(previous_index)` if it exists, otherwise `None`.
    #[inline]
    fn arena_prev_idx(&self, idx: usize) -> Option<usize> {
        let p = self.arena_prev[idx];
        // Convert sentinel to None
        if p == usize::MAX { None } else { Some(p) }
    }

    /// Returns the next alive arena index for a given slot.
    ///
    /// As with `arena_prev_idx`, this converts the `usize::MAX` sentinel into
    /// a clean `Option<usize>`. 
    ///
    /// # Arguments
    /// * `idx` – The current arena index
    ///
    /// # Returns
    /// `Some(next_index)` if it exists, otherwise `None`.
    #[inline]
    fn arena_next_idx(&self, idx: usize) -> Option<usize> {
        let n = self.arena_next[idx];
        // Convert sentinel to None
        if n == usize::MAX { None } else { Some(n) }
    }

    /// Returns the next alive arena index to the right of `left_idx`.
    ///
    /// Skips over dead slots** created during merges. 
    /// It walks the linked-list chain until it finds the next
    /// alive token or reaches the sentinel (`usize::MAX`).
    ///
    /// # Arguments
    /// * `left_idx` – The arena index whose right neighbor is requested
    ///
    /// # Returns
    /// - `Some(idx)` → the nearest alive right neighbor  
    /// - `None` → no alive tokens to the right
    #[inline]
    fn right_of(&self, left_idx: usize) -> Option<usize> {
        // Start from immediate right neighbor
        let mut n = self.arena_next[left_idx];

        // Skip dead nodes until a valid one
        while n != usize::MAX && !self.arena_alive[n] {
            n = self.arena_next[n];
        }

        // Convert sentinel to None
        if n == usize::MAX { None } else { Some(n) }
    }

    /// Returns the next alive arena index to the left of `right_idx`.
    /// 
    /// This mirrors the behavior of `right_of()`, but walks backward.
    /// Dead tokens (created during merges) are skipped automatically.
    /// The function follows the `arena_prev` chain until it finds an
    /// alive slot or reaches the sentinel.
    ///
    /// # Arguments
    /// * `right_idx` – The arena index whose left neighbor is requested
    ///
    /// # Returns
    /// - `Some(idx)` → the nearest alive left neighbor  
    /// - `None` → no alive tokens to the left
    #[inline]
    fn left_of(&self, right_idx: usize) -> Option<usize> {
        // Start from immediate left neighbor
        let mut p = self.arena_prev[right_idx];

        // Skip dead nodes moving left
        while p != usize::MAX && !self.arena_alive[p] {
            p = self.arena_prev[p];
        }

        // Convert sentinel to None
        if p == usize::MAX { None } else { Some(p) }
    }

    /// Creates a merged token by concatenating the byte sequences of tokens `a` and `b`,
    /// then interns the resulting byte sequence into the global `TokenPool`.
    ///
    /// This is used during BPE training whenever a merge operation `(a, b)` is selected.
    /// The resulting merged token gets a new unique ID assigned by the token pool,
    /// unless the same byte sequence already exists.
    ///
    /// # Arguments
    /// * `a` – ID of the left token
    /// * `b` – ID of the right token
    ///
    /// # Returns
    /// * `usize` – The token ID of the merged token (new or existing)
    fn intern_merged_token(&mut self, a: usize, b: usize) -> usize {
        // Preallocate enough space for both byte sequences
        let mut merged = Vec::with_capacity(
            self.pool.get_bytes(a).len() + self.pool.get_bytes(b).len()
        );

        // Append bytes of token `a`
        merged.extend_from_slice(self.pool.get_bytes(a));

        // Append bytes of token `b`
        merged.extend_from_slice(self.pool.get_bytes(b));

        // Intern the merged token and return its unique ID
        self.pool.intern(merged)
    }

    /// Encodes a single input string into token IDs by applying all learned BPE merges
    /// in the same deterministic order used during training, without allocating or
    /// modifying the arena-based corpus.
    ///
    /// # Arguments
    /// * `text` – The raw UTF-8 input string to encode. 
    ///
    /// # Returns
    /// * `Vec<usize>` – The sequence of token IDs produced by applying
    ///   the full BPE merge chain to the given text.
    fn encode_ids_local(&self, text: &str) -> Vec<usize> {
    // Convert text into normalized UTF-8 bytes:
    let bytes = SubwordTokenizer::preprocess_bytes(text);

    // Final output token ids for the entire input string
    let mut out: Vec<usize> = Vec::new();

    // Split into "words" by space, each word is encoded independently
    for part in bytes.split(|b| *b == b' ') {
        if part.is_empty() { continue; }

        // Each byte in the word is converted into its corresponding
        // vocabulary id (pre-interned in `TokenPool`).
        let mut syms: Vec<usize> = Vec::with_capacity(part.len());

        for &by in part {
            // key used for lookup in the pool
            let key = vec![by]; 

            // known byte symbol
            if let Some(&id) = self.pool.ids.get(&key) {
                syms.push(id);          
            } 
            // fallback to <unk> if missing 
            else {
                syms.push(0usize);      
            }
        }

        // Each merge rule (a, b) replaces consecutive symbols [a, b]
        // with the corresponding merged token, whenever found.
        for &(a, b) in &self.merges_internal {
            let mut new_syms: Vec<usize> = Vec::with_capacity(syms.len());
            let mut i = 0usize;

            while i < syms.len() {
                // Check if the pair at position i matches the merge rule
                if i + 1 < syms.len() && syms[i] == a && syms[i + 1] == b {
                    // Build merged token bytes (bytes[a] + bytes[b])
                    // and retrieve its pre-interned id.
                    let mut merged_vec = Vec::with_capacity(
                        self.pool.get_bytes(a).len() +
                        self.pool.get_bytes(b).len()
                    );
                    merged_vec.extend_from_slice(self.pool.get_bytes(a));
                    merged_vec.extend_from_slice(self.pool.get_bytes(b));

                    // Lookup merged id 
                    let mid = *self.pool.ids.get(&merged_vec).unwrap_or(&0usize);
                    
                    // Push merged token id
                    new_syms.push(mid); 
                    // Skip the consumed pair
                    i += 2;             
                } 
                else {
                    // No merge: copy current symbol as-is
                    new_syms.push(syms[i]);
                    i += 1;
                }
            }

            // The newly merged sequence becomes the working list
            syms = new_syms;
        }

        // Append encoded token ids for this word to the global output
        out.extend_from_slice(&syms);
    }

    // Return the sequence of tokens
    out
    }

    /// Converts a sequence of token IDs into their hexadecimal string representations.
    ///
    /// # Arguments
    /// * `ids` – A vector of token IDs as produced by the tokenizer.
    ///
    /// # Returns
    /// * `Vec<String>` – Each token ID is converted to a hexadecimal string representing its byte sequence.
    fn ids_to_hex_tokens(&self, ids: Vec<usize>) -> Vec<String> {
        // Map each token ID to its corresponding byte slice from the pool,
        // then convert the bytes to a hex string using the helper `bytes_to_hex`
        ids.into_iter()
            .map(|id| {
                let bytes = self.pool.get_bytes(id); 
                bytes_to_hex(bytes)                  
            })
            .collect() // collect all hex strings into a Vec
    }

    /// Converts a sequence of token IDs back into a UTF-8 string, using a lossy conversion.
    ///
    /// # Arguments
    /// * `ids` – A vector of token IDs to decode.
    ///
    /// # Returns
    /// * `String` – The decoded, human-readable string.
    fn ids_to_string_lossy(&self, ids: Vec<usize>) -> String {
        let mut out = String::new();

        for id in ids {
            // Get the byte sequence of the token from the pool
            let bytes = self.pool.get_bytes(id);

            // Convert bytes to UTF-8 string (lossy), append to output
            out.push_str(&String::from_utf8_lossy(bytes));
        }

        // Replace non-breaking spaces with regular spaces and trim surrounding whitespace
        out.replace('\u{00A0}', " ").trim().to_string()
    }
}

#[pymethods]
impl SubwordTokenizer {
    /// Creates a new `SubwordTokenizer` instance preloaded with special tokens
    /// and all single-byte tokens (0..=255).
    ///
    /// This is the constructor exposed to Python via PyO3.
    ///
    /// # Returns
    /// * `SubwordTokenizer` – A new tokenizer ready for training or encoding.
    #[new]
    pub fn new() -> Self {
        // Delegate to internal constructor that handles special tokens and single-byte pre-interning
        SubwordTokenizer::new_with_specials()
    }

    /// Trains the tokenizer on the given texts using the BPE (Byte Pair Encoding) algorithm.
    ///
    /// This function builds a subword tokenizer by iteratively merging the most frequent pairs of tokens
    /// in the input texts. It maintains an internal "arena" of token IDs and tracks which tokens are
    /// alive or already merged, building the BPE vocabulary and merge rules.
    ///
    /// # How it works
    /// 1. Resets the internal arena and word metadata.
    /// 2. Converts each input string into bytes and splits it into words.
    /// 3. Adds each word to the arena, storing token IDs and linking them via previous/next indices.
    /// 4. Counts occurrences of adjacent token pairs and stores their positions.
    /// 5. Builds a max-heap of pairs sorted by frequency.
    /// 6. Iteratively merges the most frequent pairs (up to `max_merges` or until frequency < `min_freq`),
    ///    updating the arena, counts, and heap lazily.
    /// 7. Stores the resulting merges in `self.merges_internal` and rebuilds the tokenizer's mappings.
    ///
    /// # Parameters
    /// - `&mut self`: mutable reference to the tokenizer; internal state will be updated.
    /// - `texts: Vec<String>`: the corpus of input texts used to train the tokenizer. Each string is
    ///   converted to bytes and split into words for building the arena.
    /// - `max_merges: usize`: the maximum number of merges (BPE steps) to perform. Controls the
    ///   granularity of the resulting subword vocabulary.
    /// - `min_freq: usize`: the minimum frequency for a token pair to be considered for merging.
    ///   Rare pairs are ignored to reduce noise and improve efficiency.
    ///
    /// # Returns
    /// This function returns nothing (`()`); it updates the tokenizer in-place by:
    /// - Building the internal arena of token IDs.
    /// - Updating merge rules (`self.merges_internal`).
    /// - Rebuilding the tokenizer's mappings (`self.build_mappings_from_pool()`).
    pub fn train(&mut self, texts: Vec<String>, max_merges: usize, min_freq: usize) {
        // Clear all internal data structures used for storing token IDs and word metadata
        // Prepares the tokenizer to process a new batch of texts
        
        // Vector storing the IDs of tokens in the arena is cleared
        self.arena_token_ids.clear();
    
        // Vector tracking whether each token is still "alive" (not merged) is cleared
        self.arena_alive.clear();
        
        // Vector storing the previous token index for each token is cleared
        self.arena_prev.clear(); 

        // Vector storing the next token index for each token is cleared
        self.arena_next.clear(); 

        // Vector storing the starting index of each word in the arena is cleared
        self.word_starts.clear();     

        // Vector storing the length of each word in tokens is cleared
        self.word_lens.clear();       
        
        // Initialize a counter to keep track of the current insertion index in the arena as new tokens are added
        let mut arena_index = 0usize;

        // Takes a slice of bytes (representing a word) and inserts its tokens into the arena,
        // updating all metadata vectors accordingly
        let mut add_word_to_arena = |slice: &[u8]| {
            // Do nothing if the slice is empty (avoid inserting empty words)
            if slice.is_empty() { return; } 
            
            // Record the starting index of this word in the arena
            let wstart = arena_index; 

            for &by in slice { 
                // Intern the byte in the token pool and get its unique token ID
                let id = self.pool.intern(vec![by]); 
                
                // Add the token ID to the arena vector
                self.arena_token_ids.push(id);
                // Mark this token as "alive" (not merged yet)
                self.arena_alive.push(true);   

                // Set previous token index: if this is the first token of the word, use usize::MAX to indicate "none"
                // otherwise point to the previous token in the arena.
                self.arena_prev.push(if arena_index == wstart { usize::MAX } else { arena_index - 1 });
                
                // Set next token index (will be corrected for the last token later)
                self.arena_next.push(arena_index + 1); 

                arena_index += 1; // Move the arena index forward
            }

            // Correct the next pointer of the last token in the word to indicate the end of the word
            if arena_index > wstart {
                self.arena_next[arena_index - 1] = usize::MAX;
            }
            
            // Store the starting index of the word in the word_starts vector
            self.word_starts.push(wstart);            
            // Store the length of the word in tokens
            self.word_lens.push(arena_index - wstart); 
        };

        // Iterates over each input text to convert it into tokens and store them in the arena
        for t in &texts { 
            // Preprocess the text into a byte array
            let b = SubwordTokenizer::preprocess_bytes(t);
            
            // Track the start index of the current word within the byte array
            let mut start = 0usize; 

            // Iterate through each byte in the processed text
            for (i, &ch) in b.iter().enumerate() {
                // If a space is found, treat it as a word boundary
                if ch == b' ' {
                    // Add the current word slice to the arena
                    add_word_to_arena(&b[start..i]); 
                    // Move start index to the byte after the space
                    start = i + 1; 
                }
            }

            // Add the last word in the text (after the last space or entire text if no spaces)
            add_word_to_arena(&b[start..]);
        }

        // If no words were added (empty input), just rebuild the mappings from the token pool and return early
        if self.word_starts.is_empty() {
            // Create default mappings for token IDs
            self.build_mappings_from_pool(); 
            // Exit train early since there's nothing to merge
            return; 
        }

        // Builds the frequency counts of adjacent token pairs (for BPE merges)
        // and stores the positions of each pair for efficient updates during merging

        // Define a small vector type for storing a few occurrences of (word_index, token_index)
        type Bucket = SmallVec<[(usize, usize); 4]>;

        // A hash map to count how many times each token pair appears
        let mut pair_count: FxHashMap<u64, usize> = FxHashMap::default();

        // A hash map to track where each pair occurs in the arena: (word_index, position_index)
        let mut pair_pos: FxHashMap<u64, Bucket> = FxHashMap::default();

        // Iterate over each word in the arena
        for (wid, &wstart) in self.word_starts.iter().enumerate() {
            // Length of the current word
            let wlen = self.word_lens[wid]; 

            // Skip single-token words (no pair to form)
            if wlen < 2 { continue; } 

            // Start at the first token of the word
            let mut idx = wstart; 

            // Walk through all adjacent token pairs in this word
            while let Some(next_idx) = self.arena_next_idx(idx) {
                let a = self.arena_token_ids[idx]; // Left token of the pair
                let b = self.arena_token_ids[next_idx]; // Right token of the pair

                // Combine tokens into a single u64 key for the pair
                let k = pair_key(a, b); 

                // Increment the count for this pair
                *pair_count.entry(k).or_insert(0) += 1;   
                // Record the position of this pair
                pair_pos.entry(k).or_default().push((wid, idx)); 

                // Move to the next token in the word
                idx = next_idx; 
            }
        }

        // Initialize binary heap (max-heap by count)
        let mut heap: BinaryHeap<(usize, u64)> = pair_count
            .iter()
            // Convert each pair into (count, pair_key) for the heap
            .map(|(&k, &c)| (c, k)) 
            // Collect into a heap structure
            .collect();             

        // Store the list of merged token pairs (a, b) as they are merged
        let mut merges_ids: Vec<(usize, usize)> = Vec::new();

        // Local helper function to lazily push updated counts into the heap
        fn push_current_count(
            // Heap to push updated counts into
            heap: &mut BinaryHeap<(usize, u64)>, 
            // Current frequency table of all pairs
            pair_count: &FxHashMap<u64, usize>,
            // Key of the pair to push  
            k: u64                               
        ) {
            // Only push if the pair exists and has a count
            if let Some(&c) = pair_count.get(&k) { 
                // Push (count, pair_key) into the max-heap
                heap.push((c, k));                 
            }
        }

        // Repeatedly merges the most frequent pair until >= max_merges
        while merges_ids.len() < max_merges {
            // Pop the pair with the highest frequency from the heap
            // If the heap is empty, break the loop
            let Some((cnt, key)) = heap.pop() else { break; };

            // Check the real current count in pair_count.
            let real_cnt = *pair_count.get(&key).unwrap_or(&0);
            
            if real_cnt == 0 || real_cnt != cnt { 
                // Skip if count is zero or outdated
                continue; 
            }

            // Stop merging if the pair's frequency is below the minimum threshold
            if real_cnt < min_freq { 
                break; 
            }

            // Unpack the combined u64 key into the original token IDs
            let (a, b) = unpack_pair(key);

            // Intern the new merged token in the tokenizer's pool
            let new_id = self.intern_merged_token(a, b);

            // Keep track of this merge (a, b) for updating internal mappings later
            merges_ids.push((a, b));

            // Take ownership of the occurrences bucket for this pair (if any)
            // Each entry is (word_id, left_index_in_arena)
            let occs = pair_pos.remove(&key).unwrap_or_default();

            // Iterate over all occurrences of the current pair (a, b) in the arena
            for (wid, left_idx) in occs {
                // Skip if the left token of this occurrence was already merged
                if !self.arena_alive[left_idx] { continue; }

                // Verify the right neighbor exists and still matches 'b'
                // The arena may have changed due to previous merges, so this check ensures it merge valid pairs
                if let Some(right_idx) = self.right_of(left_idx) {
                    if self.arena_token_ids[left_idx] != a || self.arena_token_ids[right_idx] != b { 
                        // Skip if tokens no longer match the pair
                        continue; 
                    }

                    // If the previous token exists, decrement the count of the pair (prev, a)
                    if let Some(ln) = self.arena_prev_idx(left_idx) {
                        // Pair key for (prev, a)
                        let lk = pair_key(self.arena_token_ids[ln], a);
                        if let Some(pc) = pair_count.get_mut(&lk) {
                            *pc = pc.saturating_sub(1); // Safely decrement, no panic on underflow
                        }
                    }
                    
                    // If the next token after 'b' exists, decrement the count of the pair (b, next)
                    if let Some(rn) = self.arena_next_idx(right_idx) {
                        // Pair key for (b, next)
                        let rk = pair_key(b, self.arena_token_ids[rn]);
                        if let Some(pc) = pair_count.get_mut(&rk) {
                            // Reduce count since the pair (b, next) will be affected by merge
                            *pc = pc.saturating_sub(1);
                        }
                    }

                    // Decrement the count of the current pair (a, b) itself
                    if let Some(pc) = pair_count.get_mut(&key) {
                        // Current pair will be merged, so its count decreases
                        *pc = pc.saturating_sub(1);
                    }

                    // Replace the left token of the pair with the new merged token ID
                    self.arena_token_ids[left_idx] = new_id;

                    // Mark the right token as dead/inactive because it is now merged
                    self.arena_alive[right_idx] = false;

                    // Get the index of the token after the right token (if any), otherwise usize::MAX indicates the end
                    let after_right = self.arena_next_idx(right_idx).unwrap_or(usize::MAX);

                    // Update the next pointer of the merged token to skip the removed right token
                    self.arena_next[left_idx] = after_right;

                    // If there is a token after the right one, update its previous pointer to point to the new merged token
                    if after_right != usize::MAX {
                        self.arena_prev[after_right] = left_idx;
                    }

                    // Check if there is a token before the merged token
                    if let Some(ln) = self.arena_prev_idx(left_idx) {
                        // Create a new pair key for (previous token, new merged token)
                        let newk = pair_key(self.arena_token_ids[ln], new_id);

                        // Increment the count of this new pair in the pair_count map
                        let entry = pair_count.entry(newk).or_insert(0);
                        *entry += 1;

                        // Record the position of this new pair for future merges
                        pair_pos.entry(newk).or_default().push((wid, ln));

                        // Lazily push the updated count into the heap to maintain max-heap property
                        push_current_count(&mut heap, &pair_count, newk);
                    }

                    // Check if there is a token after the merged token
                    if after_right != usize::MAX {
                        // Create a new pair key for (merged token, next token)
                        let newk = pair_key(new_id, self.arena_token_ids[after_right]);

                        // Increment the count of this new pair in the pair_count map
                        let entry = pair_count.entry(newk).or_insert(0);
                        *entry += 1;

                        // Record the position of this new pair for future merges
                        pair_pos.entry(newk).or_default().push((wid, left_idx));

                        // Lazily push the updated count into the heap to maintain max-heap property
                        push_current_count(&mut heap, &pair_count, newk);
                    }
                }
            }
        }

        // Save the list of all performed merges in the tokenizer's internal state
        self.merges_internal = merges_ids;

        // Rebuild the token-to-ID and ID-to-token mappings from the current pool
        // This updates all internal data structures to reflect the merged tokens
        self.build_mappings_from_pool();
    }


    /// Encodes a string into a sequence of subword tokens represented as hexadecimal strings.
    ///
    /// This method first converts the input `text` into arena-style token IDs using the BPE merges,
    /// then converts each token ID into its hexadecimal representation.
    ///
    /// # Arguments
    /// * `text` – The input string to tokenize.
    ///
    /// # Returns
    /// * `Vec<String>` – A vector of token strings in hexadecimal format.
    pub fn encode(&self, text: &str) -> Vec<String> {
        // Encode the text into local arena-style IDs
        let ids = self.encode_ids_local(text);

        // Convert token IDs to hexadecimal string tokens
        self.ids_to_hex_tokens(ids)
    }

    /// Encodes a string into a sequence of token IDs as `u32`.
    ///
    /// # Arguments
    /// * `text` – The input string to tokenize.
    ///
    /// # Returns
    /// * `Vec<u32>` – A vector of token IDs corresponding to the input text.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        // Encode the text into arena-style token IDs (usize)
        let ids = self.encode_ids_local(text);

        // Convert each usize ID to u32 for output
        ids.into_iter().map(|i| i as u32).collect()
    }

    /// Decodes a sequence of hex-encoded token strings back into a single string.
    ///
    /// Each element of `tokens_hex` should be a hex string representing a token's byte sequence.
    /// This method converts each hex string back to bytes, then interprets them as UTF-8,
    /// concatenates all tokens, replaces non-breaking spaces with normal spaces, and trims the result.
    ///
    /// # Arguments
    /// * `tokens_hex` – Vector of hex strings representing token byte sequences.
    ///
    /// # Returns
    /// * `String` – The decoded text.
    pub fn decode(&self, tokens_hex: Vec<String>) -> String {
        let mut out = String::new();

        for hx in tokens_hex {
            // Convert hex string to bytes; skip invalid ones
            if let Ok(b) = hex_to_bytes(&hx) {
                // Append UTF-8 lossy string to output
                out.push_str(&String::from_utf8_lossy(&b));
            }
        }

        // Replace non-breaking spaces with normal spaces and trim leading/trailing whitespace
        out.replace('\u{00A0}', " ").trim().to_string()
    }

    /// Decodes a sequence of token IDs into a single string.
    ///
    /// Each ID corresponds to a token in the `TokenPool`. The method looks up the byte sequence
    /// for each ID, interprets it as UTF-8, and concatenates all tokens into a single string.
    /// IDs that are out of range are replaced with the `<unk>` token.
    ///
    /// # Arguments
    /// * `ids` – Vector of token IDs (u32) to decode.
    ///
    /// # Returns
    /// * `String` – The decoded text.
    pub fn decode_from_ids(&self, ids: Vec<u32>) -> String {
        let mut out = String::new();

        for id in ids {
            let idx = id as usize;
            // If ID is valid, append corresponding token bytes as UTF-8
            if idx < self.pool.len() {
                out.push_str(&String::from_utf8_lossy(self.pool.get_bytes(idx)));
            } 
            else {
                // Out-of-range IDs mapped to <unk>
                out.push_str("<unk>");
            }
        }

        // Replace non-breaking spaces with normal spaces and trim whitespace
        out.replace('\u{00A0}', " ").trim().to_string()
    }

    /// Returns the vocabulary as a vector of hexadecimal token strings.
    ///
    /// # Returns
    /// * `Vec<String>` – Hexadecimal strings of all tokens in the pool.
    #[getter]
    pub fn vocab(&self) -> Vec<String> {
        // Convert each token's byte sequence to a hexadecimal string
        self.pool.tokens_hex()
    }

    /// Returns the list of merges as pairs of hexadecimal token strings.
    ///
    /// Each merge is represented as a tuple `(a, b)` where `a` and `b` are
    /// the hex string representations of the token byte sequences.
    /// The order of merges corresponds to the order in which they were applied
    /// during BPE training.
    ///
    /// # Returns
    /// * `Vec<(String, String)>` – List of merge pairs in hex format.
    #[getter]
    pub fn merges(&self) -> Vec<(String, String)> {
        // Convert each merge's token IDs to their corresponding hex strings
        self.merges_internal.iter().map(|(a, b)| {
            (
                bytes_to_hex(self.pool.get_bytes(*a)), // left token of the merge
                bytes_to_hex(self.pool.get_bytes(*b))  // right token of the merge
            )
        }).collect()
    }

    /// Returns the list of special tokens used by the tokenizer.
    ///
    /// Special tokens are typically reserved symbols like `<unk>`, `<pad>`,
    /// `<s>`, `</s>`, and are interned first in the token pool.
    ///
    /// # Returns
    /// * `Vec<String>` – List of special token strings.
    #[getter]
    pub fn special_tokens(&self) -> Vec<String> {
        // Clone the internal vector of special tokens to return
        self.special_tokens.clone()
    }

    /// Returns a mapping from token strings (in hex format) to their numeric IDs.
    ///
    /// # Returns
    /// * `FxHashMap<String, u32>` – Mapping from token hex string to token ID.
    #[getter]
    pub fn token_to_id(&self) -> FxHashMap<String, u32> {
        let mut out = FxHashMap::default();

        for (i, tok) in self.pool.tokens.iter().enumerate() {
            // Convert bytes to hex string and insert into map
            out.insert(bytes_to_hex(tok.as_ref()), i as u32);
        }

        out
    }

    /// Returns a vector of all tokens in the tokenizer as hex strings, indexed by their IDs.
    ///
    /// # Returns
    /// * `Vec<String>` – Tokens in hex string format, ordered by ID.
    #[getter]
    pub fn id_to_token(&self) -> Vec<String> {
        // Use the helper in TokenPool to convert each token's bytes to a hex string
        self.pool.tokens_hex()
    }

    /// Saves the current tokenizer state to a JSON file.
    ///
    /// This includes the vocabulary, merges (as pairs of token hex strings), and any special tokens. 
    ///
    /// # Arguments
    /// * `path` – File path where the tokenizer JSON will be written.
    ///
    /// # Returns
    /// * `PyResult<()>` – Returns `Ok(())` if successful, otherwise raises a Python exception.
    ///
    /// # Errors
    /// * `PyRuntimeError` if serialization fails.
    /// * `PyIOError` if writing to the file fails.
    pub fn save_to_file(&self, path: &str) -> PyResult<()> {
        // Build a serializable representation of the tokenizer
        let serial = SerializableTokenizer {
            vocab: self.pool.tokens_hex(), // convert token bytes to hex strings
            merges: self.merges_internal.iter().map(|(a,b)| {
                (
                    bytes_to_hex(self.pool.get_bytes(*a)), // left token hex
                    bytes_to_hex(self.pool.get_bytes(*b))  // right token hex
                )
            }).collect(),
            special_tokens: self.special_tokens.clone(), // copy special tokens
        };

        // Serialize to pretty JSON
        let json = serde_json::to_string_pretty(&serial)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize error: {}", e)))?;

        // Write JSON to file
        fs::write(Path::new(path), json)
            .map_err(|e| PyIOError::new_err(format!("write error: {}", e)))?;

        Ok(())
    }

    /// Loads a tokenizer from a JSON file previously saved with `save_to_file`.
    ///
    /// This reconstructs the vocabulary, merges, and special tokens.
    /// The returned `SubwordTokenizer` is ready for encoding/decoding.
    ///
    /// # Arguments
    /// * `path` – File path of the JSON file containing the tokenizer data.
    ///
    /// # Returns
    /// * `PyResult<SubwordTokenizer>` – The reconstructed tokenizer.
    ///
    /// # Errors
    /// * `PyIOError` if reading the file fails.
    /// * `PyValueError` if parsing JSON or hex values fails, or if merge tokens are missing.
    #[staticmethod]
    pub fn load_from_file(path: &str) -> PyResult<Self> {
        // Read file content
        let data = fs::read_to_string(Path::new(path))
            .map_err(|e| PyIOError::new_err(format!("read error: {}", e)))?;

        // Deserialize JSON into SerializableTokenizer
        let serial: SerializableTokenizer = serde_json::from_str(&data)
            .map_err(|e| PyValueError::new_err(format!("parse error: {}", e)))?;

        // Rebuild token pool from vocabulary
        let mut pool = TokenPool::new();
        for hx in &serial.vocab {
            let bytes = hex_to_bytes(hx)
                .map_err(|e| PyValueError::new_err(format!("hex parse: {}", e)))?;
            pool.intern(bytes);
        }

        // Convert merge pairs from hex strings to token IDs
        let mut merges_ids: Vec<(usize, usize)> = Vec::new();
        for (fa, fb) in &serial.merges {
            let a_bytes = hex_to_bytes(fa)
                .map_err(|e| PyValueError::new_err(format!("hex parse: {}", e)))?;
            let b_bytes = hex_to_bytes(fb)
                .map_err(|e| PyValueError::new_err(format!("hex parse: {}", e)))?;

            // Lookup IDs in token pool
            let a_id = *pool.ids.get(&a_bytes)
                .ok_or_else(|| PyValueError::new_err("merge token not found in vocab"))?;
            let b_id = *pool.ids.get(&b_bytes)
                .ok_or_else(|| PyValueError::new_err("merge token not found in vocab"))?;
            merges_ids.push((a_id, b_id));
        }

        // Create tokenizer instance
        let mut tok = SubwordTokenizer {
            pool,
            merges_internal: merges_ids,
            special_tokens: serial.special_tokens.clone(),
            token_to_id: FxHashMap::default(),
            id_to_token: Vec::new(),
            arena_token_ids: Vec::new(),
            arena_alive: Vec::new(),
            arena_prev: Vec::new(),
            arena_next: Vec::new(),
            word_starts: Vec::new(),
            word_lens: Vec::new(),
        };

        // Rebuild mappings for fast encoding/decoding
        tok.build_mappings_from_pool();

        Ok(tok)
    }
}
