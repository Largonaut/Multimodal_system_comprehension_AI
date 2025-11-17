#!/usr/bin/env python3
"""
GREMLIN Language Pack Generator - GUI Edition
Ultimate language pack generator with full parameter control.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import threading
from core import ConceptDictionary, WordGenerator, LanguagePack
from core.language_pack import GrammarRules


class LanguagePackGeneratorGUI:
    """Comprehensive GUI for generating GREMLIN language packs."""

    def __init__(self, root):
        self.root = root
        self.root.title("GREMLIN Language Pack Generator")
        self.root.geometry("900x800")

        # Generation state
        self.is_generating = False
        self.concept_dict = None

        # Load concept dictionary
        self.load_concept_dictionary()

        # Build UI
        self.create_widgets()

    def load_concept_dictionary(self):
        """Load the concept dictionary."""
        try:
            self.concept_dict = ConceptDictionary()
            self.total_concepts = self.concept_dict.total_concepts()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load concept dictionary: {e}")
            self.total_concepts = 186  # Fallback

    def create_widgets(self):
        """Create all UI widgets."""

        # Main container with scrolling
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0

        # ============================================================
        # HEADER
        # ============================================================
        header = ttk.Label(
            main_frame,
            text="🌍 GREMLIN Language Pack Generator 🌍",
            font=('Arial', 16, 'bold')
        )
        header.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # ============================================================
        # LANGUAGE NAME
        # ============================================================
        ttk.Label(main_frame, text="Language Name (optional):", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        self.language_name = tk.StringVar(value="")
        name_entry = ttk.Entry(main_frame, textvariable=self.language_name, width=50)
        name_entry.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        row += 1

        # ============================================================
        # WORDS PER CONCEPT
        # ============================================================
        ttk.Label(main_frame, text="Words per Concept:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        words_frame = ttk.Frame(main_frame)
        words_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        self.words_per_concept = tk.IntVar(value=10000)
        words_scale = ttk.Scale(
            words_frame,
            from_=100,
            to=50000,
            orient=tk.HORIZONTAL,
            variable=self.words_per_concept,
            command=self.update_words_label
        )
        words_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.words_label = ttk.Label(words_frame, text="10,000 words", width=15)
        self.words_label.pack(side=tk.LEFT)

        row += 1

        # Preset buttons
        preset_frame = ttk.Frame(main_frame)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Button(preset_frame, text="Tiny (500)", command=lambda: self.set_words(500)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Small (1K)", command=lambda: self.set_words(1000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Medium (5K)", command=lambda: self.set_words(5000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Large (10K)", command=lambda: self.set_words(10000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Huge (25K)", command=lambda: self.set_words(25000)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Ultra (50K)", command=lambda: self.set_words(50000)).pack(side=tk.LEFT, padx=2)

        row += 1

        # ============================================================
        # WORD LENGTH
        # ============================================================
        ttk.Label(main_frame, text="Word Length Range:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        length_frame = ttk.Frame(main_frame)
        length_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Label(length_frame, text="Min:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_length = tk.IntVar(value=4)
        ttk.Spinbox(length_frame, from_=2, to=20, textvariable=self.min_length, width=5).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(length_frame, text="Max:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_length = tk.IntVar(value=15)
        ttk.Spinbox(length_frame, from_=2, to=30, textvariable=self.max_length, width=5).pack(side=tk.LEFT)

        row += 1

        # ============================================================
        # GRAMMAR
        # ============================================================
        ttk.Label(main_frame, text="Grammar Word Order:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        grammar_frame = ttk.Frame(main_frame)
        grammar_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        self.grammar = tk.StringVar(value="SVO")
        grammars = [
            ("SVO - Subject-Verb-Object (English, Chinese)", "SVO"),
            ("SOV - Subject-Object-Verb (Japanese, Korean)", "SOV"),
            ("VSO - Verb-Subject-Object (Irish, Arabic)", "VSO"),
            ("VOS - Verb-Object-Subject (Malagasy, Fijian)", "VOS"),
            ("OVS - Object-Verb-Subject (Hixkaryana)", "OVS"),
            ("OSV - Object-Subject-Verb (Warao)", "OSV"),
        ]

        for text, value in grammars:
            ttk.Radiobutton(
                grammar_frame,
                text=text,
                variable=self.grammar,
                value=value
            ).pack(anchor=tk.W, pady=2)

        row += 1

        # ============================================================
        # UNICODE BLOCKS
        # ============================================================
        ttk.Label(main_frame, text="Unicode Character Blocks:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        # Select all/none buttons
        select_frame = ttk.Frame(main_frame)
        select_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(select_frame, text="Select All", command=self.select_all_blocks).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_frame, text="Select None", command=self.select_no_blocks).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_frame, text="Latin Only", command=self.select_latin_only).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_frame, text="Diverse Mix", command=self.select_diverse_mix).pack(side=tk.LEFT, padx=2)
        row += 1

        # Scrollable frame for Unicode blocks
        blocks_canvas = tk.Canvas(main_frame, height=150)
        blocks_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=blocks_canvas.yview)
        blocks_frame = ttk.Frame(blocks_canvas)

        blocks_canvas.configure(yscrollcommand=blocks_scrollbar.set)

        blocks_canvas.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        blocks_scrollbar.grid(row=row, column=2, sticky=(tk.N, tk.S), pady=(0, 5))

        blocks_canvas.create_window((0, 0), window=blocks_frame, anchor='nw')

        # Get Unicode blocks from WordGenerator
        wg = WordGenerator()
        available_blocks = wg.get_available_blocks()

        self.block_vars = {}
        for i, block in enumerate(available_blocks):
            var = tk.BooleanVar(value=True)  # All selected by default
            self.block_vars[block] = var

            # Format block name nicely
            display_name = wg.UNICODE_BLOCKS[block].name
            ttk.Checkbutton(
                blocks_frame,
                text=f"{display_name} ({block})",
                variable=var
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=2)

        blocks_frame.update_idletasks()
        blocks_canvas.configure(scrollregion=blocks_canvas.bbox("all"))

        row += 1

        # ============================================================
        # OUTPUT DIRECTORY
        # ============================================================
        ttk.Label(main_frame, text="Output Directory:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(15, 5)
        )
        row += 1

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        self.output_dir = tk.StringVar(value="language_packs")
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).pack(side=tk.LEFT, padx=(5, 0))

        row += 1

        # ============================================================
        # GENERATE BUTTON
        # ============================================================
        self.generate_btn = ttk.Button(
            main_frame,
            text="🚀 Generate Language Pack 🚀",
            command=self.generate_pack,
            style='Accent.TButton'
        )
        self.generate_btn.grid(row=row, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        row += 1

        # ============================================================
        # PROGRESS
        # ============================================================
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        # ============================================================
        # STATUS LOG
        # ============================================================
        ttk.Label(main_frame, text="Status:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        self.status_log = scrolledtext.ScrolledText(main_frame, height=10, width=80)
        self.status_log.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.status_log.config(state=tk.DISABLED)
        row += 1

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(row - 1, weight=1)

        # Initial status
        self.log_status("Ready to generate language packs.")
        self.log_status(f"Loaded {self.total_concepts} concepts from dictionary.")

    def update_words_label(self, value):
        """Update the words per concept label."""
        words = int(float(value))
        total = words * self.total_concepts
        file_size_mb = total * 0.042  # Rough estimate: 42 bytes per word
        self.words_label.config(
            text=f"{words:,} words\n~{total/1000:.0f}K total\n~{file_size_mb:.0f} MB"
        )

    def set_words(self, count):
        """Set words per concept to a specific value."""
        self.words_per_concept.set(count)
        self.update_words_label(count)

    def select_all_blocks(self):
        """Select all Unicode blocks."""
        for var in self.block_vars.values():
            var.set(True)

    def select_no_blocks(self):
        """Deselect all Unicode blocks."""
        for var in self.block_vars.values():
            var.set(False)

    def select_latin_only(self):
        """Select only Latin-based blocks."""
        latin_blocks = {'latin_basic', 'latin_extended_a', 'latin_extended_b'}
        for block, var in self.block_vars.items():
            var.set(block in latin_blocks)

    def select_diverse_mix(self):
        """Select a diverse mix of blocks."""
        diverse_blocks = {
            'latin_basic', 'latin_extended_a', 'cyrillic', 'greek',
            'arabic', 'hebrew', 'hiragana', 'katakana',
            'symbols_math', 'symbols_misc'
        }
        for block, var in self.block_vars.items():
            var.set(block in diverse_blocks)

    def browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)

    def log_status(self, message):
        """Log a status message."""
        self.status_log.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_log.see(tk.END)
        self.status_log.config(state=tk.DISABLED)

    def generate_pack(self):
        """Generate a language pack with current settings."""
        if self.is_generating:
            messagebox.showwarning("Busy", "Already generating a language pack.")
            return

        # Validate settings
        selected_blocks = [block for block, var in self.block_vars.items() if var.get()]
        if not selected_blocks:
            messagebox.showerror("Error", "Please select at least one Unicode block.")
            return

        min_len = self.min_length.get()
        max_len = self.max_length.get()
        if min_len >= max_len:
            messagebox.showerror("Error", "Min length must be less than max length.")
            return

        # Start generation in background thread
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED)
        self.progress.start()

        thread = threading.Thread(
            target=self._generate_pack_thread,
            args=(selected_blocks, min_len, max_len),
            daemon=True
        )
        thread.start()

    def _generate_pack_thread(self, selected_blocks, min_len, max_len):
        """Background thread for generating language pack."""
        try:
            self.log_status("=" * 60)
            self.log_status("Starting language pack generation...")

            words_per_concept = self.words_per_concept.get()
            grammar = self.grammar.get()
            output_dir = Path(self.output_dir.get())

            self.log_status(f"Words per concept: {words_per_concept:,}")
            self.log_status(f"Grammar: {grammar}")
            self.log_status(f"Word length: {min_len}-{max_len}")
            self.log_status(f"Unicode blocks: {len(selected_blocks)} selected")
            self.log_status("")

            # Create word generator
            self.log_status("Initializing word generator...")
            wg = WordGenerator(
                min_length=min_len,
                max_length=max_len,
                use_blocks=selected_blocks
            )

            # Create grammar rules
            grammar_rules = GrammarRules(word_order=grammar)

            # Generate pack
            self.log_status(f"Generating {self.total_concepts * words_per_concept:,} words...")
            self.log_status("This may take several minutes...")

            pack = LanguagePack.generate(
                concept_dict=self.concept_dict,
                words_per_concept=words_per_concept,
                grammar_rules=grammar_rules,
                word_generator=wg
            )

            # Save pack
            self.log_status("Saving language pack...")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_name = self.language_name.get().strip()
            if lang_name:
                # Sanitize language name for filename
                safe_name = "".join(c if c.isalnum() else "_" for c in lang_name)
                filename = f"language_pack_{words_per_concept}w_{grammar}_{safe_name}_{timestamp}.json"
            else:
                filename = f"language_pack_{words_per_concept}w_{grammar}_{timestamp}.json"

            output_path = output_dir / filename
            pack.save(output_path)

            # Show stats
            stats = pack.get_stats()
            file_size_mb = output_path.stat().st_size / 1024 / 1024

            self.log_status("")
            self.log_status("=" * 60)
            self.log_status("✅ Language Pack Generated Successfully! ✅")
            self.log_status("=" * 60)
            self.log_status(f"Language ID: {stats['language_id']}")
            self.log_status(f"Output File: {output_path}")
            self.log_status(f"File Size: {file_size_mb:.2f} MB")
            self.log_status(f"Total Words: {stats['total_words']:,}")
            self.log_status(f"Total Concepts: {stats['total_concepts']}")

            # Estimate authentication rounds
            words_per_exchange = 8
            estimated_rounds = stats['total_words'] // words_per_exchange
            self.log_status(f"Estimated Auth Rounds: ~{estimated_rounds:,}")
            self.log_status("=" * 60)

            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Language pack generated successfully!\n\n"
                f"File: {filename}\n"
                f"Size: {file_size_mb:.2f} MB\n"
                f"Words: {stats['total_words']:,}"
            ))

        except Exception as e:
            self.log_status(f"❌ ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Generation failed:\n{e}"))

        finally:
            # Re-enable UI
            self.root.after(0, self._generation_complete)

    def _generation_complete(self):
        """Called when generation completes."""
        self.is_generating = False
        self.generate_btn.config(state=tk.NORMAL)
        self.progress.stop()


def main():
    """Run the language pack generator GUI."""
    root = tk.Tk()

    # Configure style
    style = ttk.Style()
    style.theme_use('clam')

    app = LanguagePackGeneratorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
