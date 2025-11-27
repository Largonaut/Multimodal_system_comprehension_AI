"""
GREMLIN Tokenizer Trainer (The Librarian)
Trains a custom BPE tokenizer on the God Mode corpus.
Target: Teach the AI to treat GREMLIN codes as atomic tokens.
"""
from pathlib import Path
import sys
import json
import time

# Dependency check
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
except ImportError:
    print("Installing tokenizers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class GremlinTokenizerTrainer:
    def __init__(self):
        # Paths
        self.corpus_path = Path("training_data/gremlin_god_mode_corpus.jsonl")
        self.output_dir = Path("models/tokenizer")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Architecture Settings
        # 128k is the standard for modern LLMs (Llama 3, GPT-4).
        # It fits comfortably in 16GB VRAM.
        self.vocab_size = 128000 
        self.min_frequency = 2 # Ignore noise

    def corpus_iterator(self):
        """
        Streams ONLY the GREMLIN output column.
        The AI needs to learn to speak GREMLIN, not English (it already knows English).
        """
        if not self.corpus_path.exists():
            print(f"CRITICAL ERROR: Corpus not found at {self.corpus_path}")
            sys.exit(1)

        print(f"Reading corpus: {self.corpus_path} (This may take a moment to start)...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    yield data["output"]
                    if i % 1000000 == 0 and i > 0:
                        print(f"  Streamed {i:,} examples...")
                except:
                    continue

    def train(self):
        start_time = time.time()
        print(f"--- Initiating Tokenizer Training ---")
        print(f"Target Vocabulary: {self.vocab_size:,}")
        
        # 1. Initialize BPE (Byte Pair Encoding)
        # This is the algorithm that finds the "atomic units"
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        
        # 2. Pre-tokenizer
        # We split on ByteLevel. This ensures we can handle ANY Unicode character.
        # Essential for GREMLIN's exotic alphabet.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 3. Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # 4. The Trainer Configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            show_progress=True,
            special_tokens=[
                "<|endoftext|>", 
                "<|padding|>", 
                "<|gremlin|>", # Our custom flag
                "<|unk|>"
            ],
            # CRITICAL: We seed the alphabet with the 256 base bytes 
            # to ensure we can fallback-encode ANYTHING.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # 5. Execute Training
        print("Training started. The Librarian is reading the books...")
        tokenizer.train_from_iterator(self.corpus_iterator(), trainer=trainer)
        
        # 6. Save the Brain
        json_path = self.output_dir / "gremlin_tokenizer.json"
        tokenizer.save(str(json_path))
        
        elapsed = time.time() - start_time
        print(f"\n==================================================")
        print(f"TRAINING COMPLETE")
        print(f"Time Taken: {elapsed:.2f}s")
        print(f"Final Vocab Size: {tokenizer.get_vocab_size():,}")
        print(f"Saved to: {json_path}")
        print(f"==================================================")

if __name__ == "__main__":
    trainer = GremlinTokenizerTrainer()
    trainer.train()
