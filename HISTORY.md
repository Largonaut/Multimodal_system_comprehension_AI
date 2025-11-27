\# Project History \& Changelog



\## Phase 1: The Obfuscator

\*   \*\*Initial Goal:\*\* A "Navajo Code Talker" system for AI-to-AI authentication.

\*   \*\*Method:\*\* Random Unicode mapping.

\*   \*\*Result:\*\* Successful authentication, but inefficient.



\## Phase 2: The Pivot to Compression

\*   \*\*Discovery:\*\* Utilizing Unicode's massive address space for compression.

\*   \*\*The "0.0%" Bug:\*\* Discovered that standard Tokenizers (GPT-2) punish Unicode, hiding our gains.

\*   \*\*The Berlekamp Breakthrough:\*\* Expanded the "Alphabet" from 100 chars to 397 chars.

&nbsp;   \*   This unlocked 157,000 combinations for 2-char codes.

&nbsp;   \*   Character reduction jumped to \*\*~60%\*\*.



\## Phase 3: "Full Beans" (Scaling)

\*   \*\*The Mandate:\*\* No shortcuts. Ingest everything.

\*   \*\*Expansion:\*\* Moved from 50k words to \*\*15 Million Entries\*\*.

\*   \*\*The Dictionary Upgrade:\*\* Added NLTK Words, DWYL, and Wikipedia Titles.

\*   \*\*The Windows Bug:\*\* Hit `OverflowError` on CSV field size (32-bit limit). \*\*FIXED\*\* with dynamic sizing.



\## Phase 4: God Mode (Current State)

\*   \*\*Wikitext Integration:\*\* Fixed file pathing logic to recursively find FastAI training data.

\*   \*\*The Trap Door:\*\* Fixed a bug where Wikipedia ate all the memory slots. Implemented a "Reserve" logic to guarantee space for character fallback.

\*   \*\*Validation:\*\* Built a GUI Corpus Viewer to visually verify the "Rosetta Stone."

\*   \*\*Metrics:\*\* Updated Viewer to estimate "Project Guy Targets" (Semantic Unit Count) vs Standard Tokens.



\## Next Steps

\*   Train Custom Tokenizer (128k Vocab).

\*   Perform Model Surgery (Embedding Layer Transplant).


## Phase 5: The Librarian's Tools (Viewer v1.9)
*   **Token Metrics:** Updated Corpus Viewer to calculate true "GREMLIN Tokens" vs "Standard GPT-2 Tokens".
*   **Visuals:** Added Night Mode (Dark Theme) and Color-Coding (Green=Savings, Orange=Trap Door).
*   **Sorting:** Implemented "Pure Efficiency Sort" to identify the best compression examples (80%+ savings).
*   **Auto-Scroll:** Added Play/Pause/Speed controls to the Viewer for hands-free inspection.
*   **Tokenizer Integration:** Viewer now loads the custom `gremlin_tokenizer.json` for accurate metrics.
