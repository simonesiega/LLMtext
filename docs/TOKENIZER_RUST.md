# SubwordTokenizer — Technical Documentation (Code-Level)

This document describes the **Rust implementation** of the `SubwordTokenizer` exposed to Python via **PyO3**.  
It covers:

- Class structure  
- Internal components  
- Main public methods  
- Helper methods  
- Parameters and return values  
- Examples  
- Design rationale  

---

# 1. High-Level Overview

`SubwordTokenizer` is a Rust/Python tokenizer implementing a lightweight, efficient version of **Byte-Level BPE**.  

**Design focus:**

- **Zero-copy storage** of token byte sequences  
- **Arena-based corpus representation** for training  
- **Deterministic tokenization**  
- **Stable token IDs** stored in a `TokenPool`  
- **Fast encoding/decoding**

**Stored Components:**

| Component | Description |
|----------|-------------|
| `TokenPool` | Deduplicated store of raw token byte sequences (`Vec<u8>`) with stable IDs |
| `merges_internal` | Sequence of `(left_id, right_id)` pairs learned during BPE training |
| `token_to_id`, `id_to_token` | Runtime mappings used for encoding/decoding |
| `arena_*` | Structures used during training to efficiently merge tokens |
| `special_tokens` | `<unk>`, `<pad>`, `<s>`, `</s>` etc. |

> **Note:** Optimized for **training large corpora efficiently** with minimal memory reallocations.
---

# 2. TokenPool — Internal Token Registry

### Purpose
Stores every unique byte sequence as a token and assigns it a unique integer ID.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `ids` | `FxHashMap<Vec<u8>, usize>` | Maps byte sequence → token ID |
| `tokens` | `Vec<Arc<[u8]>>` | Maps token ID → byte sequence |

### Key Methods

- `fn new() -> TokenPool` – Creates an empty token pool  
- `fn intern(&mut self, bytes: Vec<u8>) -> usize` – Interns a byte sequence and returns its token ID  
- `fn get_bytes(&self, id: usize) -> &[u8]` – Retrieves byte sequence for a token ID  
- `fn len(&self) -> usize` – Returns number of tokens stored  
- `fn tokens_hex(&self) -> Vec<String>` – Converts tokens to hex strings  

---

# 3. Utility Functions

- `fn bytes_to_hex(b: &[u8]) -> String` – Converts bytes → hex  
- `fn hex_to_bytes(s: &str) -> Result<Vec<u8>, String>` – Converts hex → bytes  
- `pair_key(a, b)` / `unpack_pair(k)` – Encode/Decode a token pair `(a, b)` into `u64`  

---

# 4. SubwordTokenizer — Class Structure

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `pool` | `TokenPool` | Stores token bytes and IDs |
| `merges_internal` | `Vec<(usize, usize)>` | Learned BPE merges |
| `special_tokens` | `Vec<String>` | Special tokens preserved during merging |
| `token_to_id` | `FxHashMap<Arc<[u8]>, u32>` | Token → ID |
| `id_to_token` | `Vec<Arc<[u8]>>` | ID → Token |
| `arena_*` | `Vec<usize/Vec<bool>>` | Training arena |
| `word_starts` | `Vec<usize>` | Arena start index per word |
| `word_lens` | `Vec<usize>` | Length of each word |

---

# 5. Constructor

## `pub fn new() -> Self`

Python-exposed constructor. Initializes:

- Special tokens  
- All single-byte tokens (0–255)

**Python Example**

```python
from subtk import SubwordTokenizer
tok = SubwordTokenizer()
```

---

# 6. Preprocessing & Initialization Methods

- `fn preprocess_bytes(text: &str) -> Vec<u8>` – Normalize text to UTF-8 bytes  
- `fn build_mappings_from_pool(&mut self)` – Rebuild `token_to_id` and `id_to_token`

---

# 7. Arena Navigation Helpers (Training Only)

- `arena_prev_idx`, `arena_next_idx`, `right_of`, `left_of`  
- Manage linked list of tokens during training  
- Convert sentinel `usize::MAX` → `Option<usize>`  

> **Note:** Internal Use Only: Optimizes BPE merges.
---

# 8. Token Creation

## `fn intern_merged_token(&mut self, a: usize, b: usize) -> usize`

- Concatenates `token[a].bytes + token[b].bytes`  
- Interns new token in `TokenPool`  
- Returns merged token ID  

---

# 9. Encoding Methods

- `encode_ids_local(&self, text: &str) -> Vec<usize>` – Encode without modifying arena  
- `ids_to_hex_tokens(&self, ids: Vec<usize>) -> Vec<String>` – Debugging  
- `ids_to_string_lossy(&self, ids: Vec<usize>) -> String` – Decode token IDs with lossy UTF-8  

---

# 10. Train Method

## `train(&mut self, texts: Vec<String>, max_merges: usize, min_freq: usize)`

### Purpose
Builds tokenizer vocabulary and BPE merge rules from raw text data. Updates mappings and arena in-place.

### Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `texts` | `Vec<String>` | Corpus of input texts. Each string is converted to bytes and split into words |
| `max_merges` | `usize` | Maximum BPE merges to perform; controls granularity |
| `min_freq` | `usize` | Minimum frequency for token pair to be merged; rare pairs ignored |

### Return Value
`()` – Updates tokenizer in-place:

- Builds internal arena of token IDs  
- Updates `merges_internal`  
- Rebuilds `token_to_id` / `id_to_token`  

### Workflow

1. Normalize text → bytes  
2. Insert words into arena  
3. Count adjacent token pair frequencies  
4. Create max-heap by frequency  
5. Iteratively merge most frequent pairs (up to `max_merges` or frequency < `min_freq`)  
6. Update arena, pair counts, and heap lazily  
7. Store merged token pairs in `merges_internal`  
8. Rebuild token mappings  

---

### Python Wrapper Example

```python
from subtk import SubwordTokenizer

tokenizer = SubwordTokenizer()

corpus = [
    "hello world",
    "hello there",
    "world of tokenization"
]

tokenizer.train(corpus, max_merges=100, min_freq=3)
ids = tokenizer.encode("hello world")
print(ids)
```

- Python passes corpus → Rust `Vec<String>`  
- Rust performs BPE training  
- `encode()` uses learned merge rules  

---

> **Note:**
> - Training is destructive: re-running `train()` rebuilds mappings  
> - Corpus should match expected text domain  
> - Larger corpora → more stable merges  
> - `vocab_size` ≥ 256 (base byte tokens); typical: 2k–32k

