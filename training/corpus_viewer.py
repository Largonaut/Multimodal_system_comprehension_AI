"""
GREMLIN Corpus Viewer
A tool to inspect the massive God Mode training data.
VERSION: 1.8 (Auto-Scroll + Weighted Sort)
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import random
from pathlib import Path
import sys
import os
import re
import threading

# Add parent dir to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from compression.token_cost_analyzer import TokenCostAnalyzer
except ImportError:
    TokenCostAnalyzer = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

class CorpusViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GREMLIN Corpus Viewer - God Mode")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")
        
        # STYLES
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="#ffffff")
        style.configure("TButton", background="#404040", foreground="#ffffff", borderwidth=1)
        style.map("TButton", background=[('active', '#505050')])
        style.configure("TLabelframe", background="#2b2b2b", foreground="#ffffff")
        style.configure("TLabelframe.Label", background="#2b2b2b", foreground="#ffffff")
        style.configure("Horizontal.TProgressbar", background="#00ff00", troughcolor="#404040", bordercolor="#2b2b2b")
        
        # PATHS
        base_dir = Path(__file__).parent.parent
        self.corpus_path = base_dir / "training_data" / "gremlin_god_mode_corpus.jsonl"
        self.custom_tokenizer_path = base_dir / "models" / "tokenizer" / "gremlin_tokenizer.json"
        
        # STATE
        self.file_handle = None
        self.total_lines = 0
        self.offsets = [] 
        self.is_sorted = False
        self.is_playing = False
        self.play_delay = 500 # ms
        
        # ANALYZERS
        self.analyzer = None
        self.custom_tokenizer = None
        self.tokenizer_loaded = False
        
        if TokenCostAnalyzer:
            self.analyzer = TokenCostAnalyzer()
        
        self.create_layout()
        
        # STARTUP
        self.root.after(100, self.load_tokenizers_lazy)
        self.root.after(200, self.start_indexing_thread)

    def load_tokenizers_lazy(self):
        self.status_lbl.config(text="Loading Tokenizers...")
        self.root.update()
        try:
            if self.analyzer:
                self.analyzer.load_tokenizer()
            
            if Tokenizer and self.custom_tokenizer_path.exists():
                self.custom_tokenizer = Tokenizer.from_file(str(self.custom_tokenizer_path))
                print(f"Loaded Custom Tokenizer from {self.custom_tokenizer_path}")
            
            self.tokenizer_loaded = True
            self.status_lbl.config(text="Tokenizers loaded.")
        except Exception as e:
            print(f"Tokenizer error: {e}")

    def create_layout(self):
        # Header Frame
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)
        
        # LEFT: Status & Progress
        ctrl_frame = ttk.Frame(header_frame)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.status_lbl = ttk.Label(ctrl_frame, text="Status: Initializing...", font=("Arial", 10))
        self.status_lbl.pack(side=tk.TOP, anchor=tk.W)
        
        self.progress = ttk.Progressbar(ctrl_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(side=tk.TOP, anchor=tk.W, fill=tk.X, pady=(5,0))
        
        # RIGHT: Controls
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        # Sorting
        ttk.Button(btn_frame, text="Sort by Savings ⬇", command=self.start_sort_thread).pack(side=tk.LEFT, padx=(0, 20))
        
        # Playback Controls
        play_frame = ttk.LabelFrame(btn_frame, text="Auto-Scroll", padding=5)
        play_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.btn_play = ttk.Button(play_frame, text="▶ Play", width=6, command=self.toggle_playback)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(play_frame, text="Speed:").pack(side=tk.LEFT, padx=2)
        self.scale_speed = tk.Scale(play_frame, from_=10, to=1000, orient=tk.HORIZONTAL, showvalue=0, 
                                    bg="#2b2b2b", fg="white", troughcolor="#404040", highlightthickness=0, length=100)
        self.scale_speed.set(500) # Default delay
        self.scale_speed.bind("<Motion>", self.update_speed)
        self.scale_speed.pack(side=tk.LEFT, padx=5)
        self.lbl_speed = ttk.Label(play_frame, text="Med")
        self.lbl_speed.pack(side=tk.LEFT)

        # Navigation
        ttk.Button(btn_frame, text="First", width=5, command=lambda: self.goto_line(0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Prev", width=5, command=self.prev_line).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Random", width=8, command=self.random_line).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Next", width=5, command=self.next_line).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Last", width=5, command=lambda: self.goto_line(self.total_lines-1)).pack(side=tk.LEFT, padx=1)

        # Content Area
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        self.source_lbl = ttk.Label(content_frame, text="Source: -", font=("Arial", 12, "bold"))
        self.source_lbl.pack(anchor=tk.W, pady=(0,10))

        # Input
        input_frame = ttk.LabelFrame(content_frame, text="Input (English)", padding=5)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        self.lbl_input_metric = ttk.Label(input_frame, text="Standard Tokens (GPT-2): -", foreground="#aaaaaa")
        self.lbl_input_metric.pack(anchor=tk.E)
        self.txt_input = tk.Text(input_frame, height=8, wrap=tk.WORD, font=("Arial", 11), bg="#404040", fg="#ffffff", insertbackground="white")
        self.txt_input.pack(fill=tk.X)
        
        # Output
        output_frame = ttk.LabelFrame(content_frame, text="Output (GREMLIN)", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.lbl_output_metric = ttk.Label(output_frame, text="Actual GREMLIN Tokens: -", foreground="#00ff00", font=("Arial", 10, "bold"))
        self.lbl_output_metric.pack(anchor=tk.E)
        self.txt_output = tk.Text(output_frame, height=8, wrap=tk.WORD, font=("Arial", 14), bg="#1e1e1e", fg="#00ff00", insertbackground="white")
        self.txt_output.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        jump_frame = ttk.Frame(self.root, padding=10)
        jump_frame.pack(fill=tk.X)
        ttk.Label(jump_frame, text="Jump to Line:").pack(side=tk.LEFT)
        self.ent_jump = ttk.Entry(jump_frame, width=10)
        self.ent_jump.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_entry).pack(side=tk.LEFT)
        self.lbl_pos = ttk.Label(jump_frame, text="Line: 0 / 0")
        self.lbl_pos.pack(side=tk.RIGHT)

        self.current_line_idx = 0

    # --- PLAYBACK LOGIC ---
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause")
            self.auto_play_loop()
        else:
            self.btn_play.config(text="▶ Play")

    def update_speed(self, event):
        # Invert scale: Low value = Fast speed (Low delay)
        # Scale 10 (Fast) -> 1000 (Slow)
        val = self.scale_speed.get()
        self.play_delay = val
        
        if val < 100: txt = "Hyper"
        elif val < 300: txt = "Fast"
        elif val < 600: txt = "Med"
        else: txt = "Slow"
        self.lbl_speed.config(text=txt)

    def auto_play_loop(self):
        if not self.is_playing: return
        
        self.next_line()
        
        # Schedule next frame
        self.root.after(self.play_delay, self.auto_play_loop)

    # --- INDEXING LOGIC ---
    def start_indexing_thread(self):
        threading.Thread(target=self.index_file, daemon=True).start()

    def index_file(self):
        if not self.corpus_path.exists():
            self.root.after(0, lambda: self.status_lbl.config(text="Error: Corpus file not found!", foreground="red"))
            return

        self.root.after(0, lambda: self.status_lbl.config(text="Indexing 2GB+ file..."))
        
        file_size = self.corpus_path.stat().st_size
        offsets = [0]
        try:
            offset = 0
            processed_bytes = 0
            with open(self.corpus_path, "rb") as f:
                for line in f:
                    length = len(line)
                    offset += length
                    offsets.append(offset)
                    processed_bytes += length
                    if len(offsets) % 100000 == 0:
                        pct = (processed_bytes / file_size) * 100
                        self.root.after(0, lambda p=pct: self.progress.configure(value=p))
                        self.root.after(0, lambda c=len(offsets): self.status_lbl.config(text=f"Indexing... {c:,} lines"))

            if len(offsets) > 1: offsets.pop()
            self.total_lines = len(offsets)
            self.offsets = offsets
            
            self.root.after(0, lambda: self.finish_indexing(f"Ready. Loaded {self.total_lines:,} lines."))
        except Exception as e:
            self.root.after(0, lambda: self.status_lbl.config(text=f"Error indexing: {e}", foreground="red"))

    def finish_indexing(self, msg):
        self.status_lbl.config(text=msg)
        self.progress.configure(value=0)
        if self.total_lines > 0: self.goto_line(0)

    # --- SORTING LOGIC ---
    def start_sort_thread(self):
        if self.is_sorted:
            messagebox.showinfo("Info", "Already sorted by savings.")
            return
        if self.total_lines == 0: return
        threading.Thread(target=self.sort_by_savings, daemon=True).start()

    def sort_by_savings(self):
        self.root.after(0, lambda: self.status_lbl.config(text="Deep Scanning for High Scores (~2 mins)..."))
        scored_offsets = []
        count = 0
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for i, offset in enumerate(self.offsets):
                    f.seek(offset)
                    line = f.readline()
                    if line:
                        try:
                            data = json.loads(line)
                            inp = data.get('input', '')
                            out = data.get('output', '')
                            
                            # WEIGHTED SCORE
                            # 1. Basic Ratio: 1 - (out/inp)
                            # 2. Penalty for tiny inputs (prevents "a" -> "b" being top rank)
                            if len(inp) > 2:
                                ratio = 1 - (len(out) / len(inp))
                                # Penalty factor: Approaches 1.0 as string gets longer
                                # A 3 char string gets 0.33 penalty. A 100 char string gets 0.99.
                                penalty = 1.0 - (1.0 / len(inp))
                                score = ratio * penalty
                            else:
                                score = -100 # Bury tiny strings
                            
                            scored_offsets.append((offset, score, len(inp)))
                        except:
                            scored_offsets.append((offset, -100, 0))
                    count += 1
                    if count % 100000 == 0:
                        pct = (count / self.total_lines) * 100
                        self.root.after(0, lambda p=pct: self.progress.configure(value=p))
                        self.root.after(0, lambda c=count: self.status_lbl.config(text=f"Scoring... {c:,}"))

            self.root.after(0, lambda: self.status_lbl.config(text="Sorting results..."))
            
            # Sort Key: (Score Descending, Length Descending)
            scored_offsets.sort(key=lambda x: (x[1], x[2]), reverse=True)
            self.offsets = [x[0] for x in scored_offsets]
            self.is_sorted = True
            self.root.after(0, lambda: self.finish_indexing("Sorted by Weighted Savings!"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_lbl.config(text=f"Sort Error: {e}", foreground="red"))

    # --- DISPLAY LOGIC ---
    def calculate_std_tokens(self, text):
        if not self.analyzer or not self.analyzer.tokenizer: return 0
        try: return len(self.analyzer.tokenizer.encode(text))
        except: return 0

    def calculate_custom_tokens(self, text):
        if not self.custom_tokenizer: return 0
        try: return len(self.custom_tokenizer.encode(text).ids)
        except: return 0

    def load_line(self, idx):
        if self.total_lines == 0: return
        if idx < 0: idx = 0
        if idx >= self.total_lines: idx = self.total_lines - 1
        
        self.current_line_idx = idx
        offset = self.offsets[idx]
        
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                line = f.readline()
                if line:
                    data = json.loads(line)
                    self.display_data(data)
                    self.lbl_pos.config(text=f"Line: {idx+1:,} / {self.total_lines:,}")
        except Exception as e:
            print(f"Error loading line {idx}: {e}")

    def display_data(self, data):
        source = data.get('source', 'Unknown').upper()
        inp_text = data.get('input', '')
        out_text = data.get('output', '')
        
        self.source_lbl.config(text=f"Source: {source}")
        
        self.txt_input.delete("1.0", tk.END)
        self.txt_input.insert("1.0", inp_text)
        
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert("1.0", out_text)
        
        if self.tokenizer_loaded:
            in_tok = self.calculate_std_tokens(inp_text)
            gremlin_tok = self.calculate_custom_tokens(out_text)
            
            pct = 0.0
            if in_tok > 0:
                pct = (1 - (gremlin_tok / in_tok)) * 100
            
            self.lbl_input_metric.config(text=f"Standard Tokens (GPT-2): {in_tok}")
            
            color = "#00ff00" if pct > 0 else "#ffaa00"
            self.lbl_output_metric.configure(foreground=color)
            self.lbl_output_metric.config(
                text=f"Standard: {in_tok} ➔ GREMLIN: {gremlin_tok} ({pct:.1f}% Savings)"
            )

    def next_line(self): self.load_line(self.current_line_idx + 1)
    def prev_line(self): self.load_line(self.current_line_idx - 1)
    def goto_line(self, idx): self.load_line(idx)
    def random_line(self): 
        if self.total_lines > 0: self.load_line(random.randint(0, self.total_lines - 1))
    def jump_to_entry(self):
        try:
            idx = int(self.ent_jump.get()) - 1
            self.load_line(idx)
        except ValueError: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CorpusViewer(root)
    root.mainloop()
