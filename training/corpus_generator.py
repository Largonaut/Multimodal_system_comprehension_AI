"""
GREMLIN God Mode Corpus Generator
Generates the 'Rosetta Stone' training data.
UPDATED: Loads existing Dictionary from Vault (compression/dictionaries).
"""
from pathlib import Path
import sys
import random
import json
import time
import re
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))
from compression.semantic_compression_dict import SemanticCompressionDict

class GodModeCorpusGenerator:
    def __init__(self):
        print("Initializing God Mode Generator...")
        self.builder = SemanticCompressionDict()
        
        # VAULT LOGIC: Find latest dictionary
        vault_dir = Path("compression/dictionaries")
        latest_dict = None
        
        if vault_dir.exists():
            files = list(vault_dir.glob("*.json"))
            if files:
                latest_dict = max(files, key=lambda p: p.stat().st_mtime)
        
        if latest_dict:
            print(f"Found Dictionary in Vault: {latest_dict.name}")
            success = self.builder.load(latest_dict)
            if not success:
                print("Load failed. Rebuilding from scratch...")
                self.builder.build(full_beans=True)
        else:
            print("No dictionary found in Vault. Building from scratch (High RAM usage)...")
            self.builder.build(full_beans=True)
        
        self.output_dir = Path("training_data")
        self.output_dir.mkdir(exist_ok=True)
        self.output_file = self.output_dir / "gremlin_god_mode_corpus.jsonl"
        
        csv.field_size_limit(2147483647)

    def compress_text(self, text):
        compressed_parts = []
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.builder.compression_map:
                compressed_parts.append(self.builder.compression_map[lower_token])
            else:
                for char in lower_token:
                    if char in self.builder.char_map:
                        compressed_parts.append(self.builder.char_map[char])
                    else:
                        compressed_parts.append(char)
        return "".join(compressed_parts)

    def process_corpus(self, name, sentence_iterator, f):
        print(f"\n[Stream] Processing {name} Corpus...")
        count = 0
        buffer = []
        chunk_size = random.randint(2, 6)
        for sent in sentence_iterator:
            text = " ".join(sent) if isinstance(sent, list) else sent
            buffer.append(text)
            if len(buffer) >= chunk_size:
                original = " ".join(buffer)
                if len(original) > 50:
                    compressed = self.compress_text(original)
                    f.write(json.dumps({"source": name.lower(), "input": original, "output": compressed}, ensure_ascii=False) + "\n")
                    count += 1
                buffer = []
                chunk_size = random.randint(2, 6)
        print(f"  ✓ Added {count:,} samples from {name}.")
        return count

    def find_wikitext_file(self):
        base_path = Path("ingestion/data")
        print(f"  Searching for Wikitext in {base_path.absolute()}...")
        patterns = ["train.csv", "wiki.train.tokens", "wiki.train.raw", "*train.tokens"]
        for pattern in patterns:
            matches = list(base_path.rglob(pattern))
            if matches: return matches[0]
        return None

    def process_local_wikitext(self, f):
        print("\n[Stream] Searching for Wikitext-103...")
        wiki_path = self.find_wikitext_file()
        if not wiki_path:
            print(f"  ! Error: Wikitext file not found.")
            return 0

        print(f"  Found: {wiki_path}")
        print("  Processing (This is the heavy lifting)...")
        
        count = 0
        is_csv = wiki_path.suffix.lower() == '.csv'
        
        try:
            with open(wiki_path, "r", encoding="utf-8", errors="ignore") as w:
                if is_csv:
                    reader = csv.reader(w)
                    try:
                        for row in reader:
                            if not row: continue
                            text = row[-1]
                            if len(text) < 50: continue
                            compressed = self.compress_text(text)
                            f.write(json.dumps({"source": "wikitext", "input": text, "output": compressed}, ensure_ascii=False) + "\n")
                            count += 1
                            if count % 50000 == 0: print(f"  ... processed {count:,} articles")
                    except csv.Error as e: print(f"  ! CSV Error: {e}")
                else:
                    for line in w:
                        line = line.strip().replace(" <unk>", "").replace(" @-@ ", "-")
                        if not line or line.startswith(" =") or len(line) < 50: continue
                        compressed = self.compress_text(line)
                        f.write(json.dumps({"source": "wikitext", "input": line, "output": compressed}, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 50000 == 0: print(f"  ... processed {count:,} articles")
            print(f"  ✓ Added {count:,} Wikipedia segments.")
            return count
        except Exception as e:
            print(f"  ! Error processing Wikitext: {e}")
            return 0

    def generate_flashcards(self, f):
        print("\n[Stream] Generating Universal Flashcards...")
        keys = list(self.builder.compression_map.keys())
        print(f"  Target: {len(keys):,} entries.")
        count = 0
        for key in keys:
            if key.startswith("wordnet_") or len(key) > 5:
                templates = [f"The term is {key}.", f"Define {key}.", f"{key}", f"Input: {key}"]
                original = random.choice(templates)
                compressed = self.compress_text(original)
                f.write(json.dumps({"source": "flashcard", "input": original, "output": compressed}, ensure_ascii=False) + "\n")
                count += 1
                if count % 250000 == 0: print(f"  ... {count:,} generated")
        print(f"  ✓ Added {count:,} flashcards.")
        return count

    def run(self):
        print(f"--- Starting God Mode Corpus Generation ---")
        print(f"Output: {self.output_file}")
        start_time = time.time()
        total_records = 0

        with open(self.output_file, "w", encoding="utf-8") as f:
            try:
                from nltk.corpus import brown, reuters, gutenberg
                total_records += self.process_corpus("Brown", brown.sents(), f)
                try: total_records += self.process_corpus("Reuters", reuters.sents(), f)
                except: pass
                try: total_records += self.process_corpus("Gutenberg", gutenberg.sents(), f)
                except: pass
            except: pass

            total_records += self.process_local_wikitext(f)

            try:
                from nltk.corpus import wordnet as wn
                print("\n[Stream] Processing WordNet Definitions...")
                c = 0
                for syn in wn.all_synsets():
                    orig = syn.definition()
                    comp = self.compress_text(orig)
                    f.write(json.dumps({"source": "wordnet", "input": orig, "output": comp}, ensure_ascii=False)+"\n")
                    c += 1
                print(f"  ✓ Added {c:,} definitions.")
                total_records += c
            except: pass

            total_records += self.generate_flashcards(f)

        elapsed = time.time() - start_time
        print(f"\n========================================")
        print(f"GENERATION COMPLETE")
        print(f"Total Training Pairs: {total_records:,}")
        print(f"Time Taken: {elapsed:.2f}s")
        print(f"File Size: {self.output_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"========================================")

if __name__ == "__main__":
    gen = GodModeCorpusGenerator()
    gen.run()
