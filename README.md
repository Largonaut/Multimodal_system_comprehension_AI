# GREMLIN: Semantic Compression & Obfuscation Engine
**Current State:** GOD MODE (v1.5)

GREMLIN is a **Semantic One-Time Pad** and **Hyper-Compression Engine**. It maps the entirety of the English language and human knowledge (Wikipedia) into a dense, 3-character Unicode vocabulary.

It is designed as the foundational communication and memory layer for **Project Guy**.

---

## 🚀 Current Capabilities

### 1. Massive Semantic Dictionary ("God Mode")
*   **Capacity:** 15 Million entries.
*   **Coverage:** 
    *   **Lexical:** Brown Corpus, NLTK Words, DWYL (~466k words).
    *   **Semantic:** WordNet Concepts (~117k synsets).
    *   **Entities:** English Wikipedia Titles (~7M+ entities).
*   **The Trap Door:** A fallback character-encoding mechanism ensuring **100.0% coverage**. No word is ever left in plain text.

### 2. Universal Training Corpus
*   **Size:** ~2.3 GB JSONL file.
*   **Content:** ~15 Million training pairs mapping English -> GREMLIN.
*   **Sources:** Brown, Reuters, Gutenberg, Wikitext-103, and Dictionary Flashcards.
*   **Purpose:** The "Rosetta Stone" used to train the Custom Tokenizer and AI Model.

### 3. Tooling Suite
*   **Ingestion Engine:** Automatically fetches and repairs raw datasets (FastAI Wikitext mirror).
*   **Dictionary Viewer:** GUI to inspect the 15M-entry hash map.
*   **Corpus Viewer:** GUI with Auto-Scroll, Efficiency Sorting, and Live Token Metrics.
*   **Compression GUI:** Dashboard for testing compression rates on custom text.

---

## 🛠️ Usage Guide (Python 3.12)

### 1. Data Pipeline (Rebuild from Scratch)
```bash
# 1. Download raw data (Wikidata, Word lists, Corpora)
py -3.12 ingestion/fetch_data.py

# 2. Generate the Dictionary & Training Corpus (The Heavy Lift)
# WARNING: Requires ~8-16GB RAM. Takes ~5-10 minutes.
py -3.12 training/corpus_generator.py
2. Inspection Tools
code
Bash
# View the Training Data & Metrics
py -3.12 training/corpus_viewer.py

# Inspect the Dictionary (requires building it first inside the tool)
py -3.12 compression/dictionary_viewer.py

# Test Compression on Custom Text
py -3.12 compression/compression_gui.py
3. The "Forge" (Next Step)
code
Bash
# Train the Custom Tokenizer on the Corpus
py -3.12 training/train_tokenizer.py
📊 Performance Metrics
Character Reduction: ~55% - 80% (depending on content density).
Token Reduction: pending custom tokenizer training (Expected ~2x-4x multiplier).
Security: 100% Obfuscation (Semantic OTP).
📂 Project Structure
compression/ - Core logic for Dictionary Building and Token Analysis.
ingestion/ - Scripts to fetch raw internet data.
training/ - Generators for the massive JSONL datasets and Tokenizer training.
core/ - WordNet extraction logic.