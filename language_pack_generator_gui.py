#!/usr/bin/env python3
"""
GREMLIN Language Pack Generator - GUI Edition
Ultimate language pack generator with full parameter control and WordNet integration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import threading
from core import ConceptDictionary, WordGenerator, LanguagePack
from core.language_pack import GrammarRules
from core.wordnet_concepts import WordNetConceptExtractor


# Tier configurations
CONCEPT_TIERS = {
    'base': {
        'name': 'Base Concepts (186)',
        'count': 186,
        'description': 'Core authentication and command concepts',
        'recommended_words': 10000,
        'max_words': 50000
    },
    'wordnet_basic': {
        'name': 'WordNet - Basic (3,000 synsets)',
        'count': 3000,
        'description': 'Conversational fluency vocabulary',
        'recommended_words': 1000,
        'max_words': 2000
    },
    'wordnet_standard': {
        'name': 'WordNet - Standard (10,000 synsets)',
        'count': 10000,
        'description': 'Educated adult vocabulary',
        'recommended_words': 1000,
        'max_words': 2000
    },
    'wordnet_professional': {
        'name': 'WordNet - Professional (20,000 synsets)',
        'count': 20000,
        'description': 'Large vocabulary individual',
        'recommended_words': 500,
        'max_words': 2000
    },
    'wordnet_complete': {
        'name': 'WordNet - Complete (117,000+ synsets)',
        'count': 117659,
        'description': 'FULL WordNet corpus (Dev Mode)',
        'recommended_words': 500,
        'max_words': 20000  # Dev mode unlocks this
    }
}


class LanguagePackGeneratorGUI:
    """Comprehensive GUI for generating GREMLIN language packs with WordNet support."""

    def __init__(self, root):
        self.root = root
        self.root.title("GREMLIN Language Pack Generator - WordNet Edition")
        self.root.geometry("950x900")

        # Generation state
        self.is_generating = False
        self.concept_dict = None
        self.wordnet_extractor = None

        # Current tier
        self.current_tier = 'base'

        # Load base concept dictionary
        self.load_base_concepts()

        # Build UI
        self.create_widgets()

    def load_base_concepts(self):
        """Load the base concept dictionary."""
        try:
            self.concept_dict = ConceptDictionary()
            print(f"✅ Loaded {self.concept_dict.total_concepts()} base concepts")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load concept dictionary: {e}")

    def create_widgets(self):
        """Create all UI widgets."""

        # Main container with scrolling
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = scrollable_frame

        row = 0

        # ============================================================
        # HEADER
        # ============================================================
        header = ttk.Label(
            main_frame,
            text="🌍 GREMLIN Language Pack Generator 🌍",
            font=('Arial', 16, 'bold')
        )
        header.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        subtitle = ttk.Label(
            main_frame,
            text="WordNet Edition - Up to 117,000 synsets",
            font=('Arial', 10, 'italic')
        )
        subtitle.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # ============================================================
        # CONCEPT SOURCE (TIERS)
        # ============================================================
        ttk.Label(main_frame, text="Concept Source:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        tier_frame = ttk.Frame(main_frame)
        tier_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        self.tier_var = tk.StringVar(value='base')
        tier_dropdown = ttk.Combobox(
            tier_frame,
            textvariable=self.tier_var,
            values=[t['name'] for t in CONCEPT_TIERS.values()],
            state='readonly',
            width=50
        )
        tier_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        tier_dropdown.bind('<<ComboboxSelected>>', self.on_tier_changed)

        self.tier_info_label = ttk.Label(tier_frame, text="", foreground='blue')
        self.tier_info_label.pack(side=tk.LEFT)

        row += 1

        # Tier description
        self.tier_desc_label = ttk.Label(
            main_frame,
            text="Core authentication and command concepts",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        )
        self.tier_desc_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
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
        # WORDS PER SYNSET
        # ============================================================
        ttk.Label(main_frame, text="Words per Synset:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        words_frame = ttk.Frame(main_frame)
        words_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        self.words_per_concept = tk.IntVar(value=10000)
        self.words_scale = ttk.Scale(
            words_frame,
            from_=100,
            to=50000,
            orient=tk.HORIZONTAL,
            variable=self.words_per_concept,
            command=self.update_words_label
        )
        self.words_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.words_label = ttk.Label(words_frame, text="", width=25)
        self.words_label.pack(side=tk.LEFT)

        row += 1

        # Dev mode checkbox
        dev_frame = ttk.Frame(main_frame)
        dev_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        self.dev_mode = tk.BooleanVar(value=False)
        dev_check = ttk.Checkbutton(
            dev_frame,
            text="🔓 Dev Mode (unlock up to 20,000 words/synset for Complete tier)",
            variable=self.dev_mode,
            command=self.on_dev_mode_changed
        )
        dev_check.pack(side=tk.LEFT)

        row += 1

        # Preset buttons (will update based on tier)
        self.preset_frame = ttk.Frame(main_frame)
        self.preset_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        self.create_preset_buttons()

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
        blocks_canvas = tk.Canvas(main_frame, height=120)
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
        # FILE SIZE ESTIMATE
        # ============================================================
        self.estimate_label = ttk.Label(
            main_frame,
            text="",
            font=('Arial', 9, 'bold'),
            foreground='green'
        )
        self.estimate_label.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        # ============================================================
        # GENERATE BUTTON
        # ============================================================
        self.generate_btn = ttk.Button(
            main_frame,
            text="🚀 Generate Language Pack 🚀",
            command=self.generate_pack
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

        self.status_log = scrolledtext.ScrolledText(main_frame, height=12, width=80)
        self.status_log.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.status_log.config(state=tk.DISABLED)
        row += 1

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)

        # Initial UI state
        self.update_ui_for_tier()
        self.update_words_label(self.words_per_concept.get())

        # Initial status
        self.log_status("Ready to generate language packs.")
        self.log_status(f"Base concepts loaded: {CONCEPT_TIERS['base']['count']}")
        self.log_status(f"WordNet available: {CONCEPT_TIERS['wordnet_complete']['count']} synsets")

    def create_preset_buttons(self):
        """Create preset buttons based on current tier."""
        # Clear existing buttons
        for widget in self.preset_frame.winfo_children():
            widget.destroy()

        tier_key = self.get_tier_key_from_name(self.tier_var.get())
        tier = CONCEPT_TIERS[tier_key]

        # Presets based on tier
        if tier_key == 'base':
            presets = [(500, "Tiny"), (1000, "Small"), (5000, "Medium"),
                      (10000, "Large"), (25000, "Huge"), (50000, "Ultra")]
        else:
            # WordNet tiers - smaller presets for portability
            presets = [(100, "Micro"), (500, "Small"), (1000, "Medium"),
                      (1500, "Large"), (2000, "Max")]

        for words, label in presets:
            if words <= tier['max_words']:
                ttk.Button(
                    self.preset_frame,
                    text=f"{label} ({words})",
                    command=lambda w=words: self.set_words(w)
                ).pack(side=tk.LEFT, padx=2)

    def get_tier_key_from_name(self, name):
        """Get tier key from display name."""
        for key, tier in CONCEPT_TIERS.items():
            if tier['name'] == name:
                return key
        return 'base'

    def on_tier_changed(self, event=None):
        """Handle tier selection change."""
        self.current_tier = self.get_tier_key_from_name(self.tier_var.get())
        self.update_ui_for_tier()

    def on_dev_mode_changed(self):
        """Handle dev mode toggle."""
        self.update_ui_for_tier()

    def update_ui_for_tier(self):
        """Update UI based on selected tier."""
        tier = CONCEPT_TIERS[self.current_tier]

        # Update description
        self.tier_desc_label.config(text=tier['description'])

        # Update tier info
        self.tier_info_label.config(text=f"{tier['count']:,} concepts")

        # Update slider max
        if self.current_tier == 'wordnet_complete' and self.dev_mode.get():
            max_words = 20000
            self.log_status("🔓 Dev Mode: Unlocked 20,000 words/synset for Complete tier")
        else:
            max_words = tier['max_words']

        self.words_scale.config(to=max_words)

        # Set recommended words
        recommended = tier['recommended_words']
        if self.words_per_concept.get() > max_words:
            self.words_per_concept.set(recommended)

        # Recreate preset buttons
        self.create_preset_buttons()

        # Update estimate
        self.update_words_label(self.words_per_concept.get())

    def update_words_label(self, value):
        """Update the words per synset label and file size estimate."""
        words = int(float(value))
        tier = CONCEPT_TIERS[self.current_tier]
        total_words = words * tier['count']
        file_size_mb = total_words * 0.042  # Rough estimate: 42 bytes per word

        self.words_label.config(
            text=f"{words:,} words\n~{total_words/1000:.0f}K total"
        )

        # File size estimate with color coding
        if file_size_mb < 100:
            color = 'green'
            status = "✅ Portable"
        elif file_size_mb < 1000:
            color = 'orange'
            status = "⚠️ Large"
        else:
            color = 'red'
            status = "🔴 Very Large"

        # Estimate generation time
        if total_words < 200000:
            time_est = "~1-2 min"
        elif total_words < 1000000:
            time_est = "~2-5 min"
        elif total_words < 10000000:
            time_est = "~10-30 min"
        else:
            time_est = "~1-8 hours"

        self.estimate_label.config(
            text=f"{status}  File Size: ~{file_size_mb:.1f} MB  |  Est. Time: {time_est}  |  Total Words: {total_words:,}",
            foreground=color
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

        # Get tier info
        tier = CONCEPT_TIERS[self.current_tier]
        words_per_concept = self.words_per_concept.get()
        total_words = tier['count'] * words_per_concept
        file_size_mb = total_words * 0.042

        # Warning for large packs
        if file_size_mb > 5000:
            response = messagebox.askyesno(
                "Large Pack Warning",
                f"This will create a {file_size_mb:.1f} MB file with {total_words:,} words.\n"
                f"Generation may take several hours.\n\n"
                f"Continue?"
            )
            if not response:
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

            tier = CONCEPT_TIERS[self.current_tier]
            words_per_concept = self.words_per_concept.get()
            grammar = self.grammar.get()
            output_dir = Path(self.output_dir.get())

            self.log_status(f"Tier: {tier['name']}")
            self.log_status(f"Concepts: {tier['count']:,}")
            self.log_status(f"Words per concept: {words_per_concept:,}")
            self.log_status(f"Grammar: {grammar}")
            self.log_status(f"Word length: {min_len}-{max_len}")
            self.log_status(f"Unicode blocks: {len(selected_blocks)} selected")
            self.log_status("")

            # Get concepts based on tier
            if self.current_tier == 'base':
                # Use base concepts
                concepts_source = self.concept_dict
                self.log_status("Using base concept dictionary...")
            else:
                # Extract WordNet synsets
                self.log_status(f"Extracting WordNet synsets ({tier['count']:,})...")
                self.log_status("This may take a few minutes...")

                if not self.wordnet_extractor:
                    self.wordnet_extractor = WordNetConceptExtractor()

                if self.current_tier == 'wordnet_complete':
                    # Get ALL synsets
                    concepts = self.wordnet_extractor.get_all_concepts(
                        progress_callback=self.update_extraction_progress
                    )
                else:
                    # Get top N synsets
                    concepts = self.wordnet_extractor.extract_top_synsets(
                        tier['count'],
                        progress_callback=self.update_extraction_progress
                    )

                # Create a temporary concept dictionary from WordNet concepts
                from core.concepts import ConceptDictionary as CD
                concepts_source = CD()
                concepts_source.concepts = {c.id: c for c in concepts}

                self.log_status(f"✅ Extracted {len(concepts)} concepts from WordNet")

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
            self.log_status(f"Generating {tier['count'] * words_per_concept:,} words...")
            self.log_status("This may take several minutes...")

            pack = LanguagePack.generate(
                concept_dict=concepts_source,
                words_per_concept=words_per_concept,
                grammar_rules=grammar_rules,
                word_generator=wg
            )

            # Save pack
            self.log_status("Saving language pack...")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_name = self.language_name.get().strip()

            # Include tier in filename
            tier_short = self.current_tier.replace('wordnet_', 'wn_')

            if lang_name:
                # Sanitize language name for filename
                safe_name = "".join(c if c.isalnum() else "_" for c in lang_name)
                filename = f"language_pack_{words_per_concept}w_{grammar}_{tier_short}_{safe_name}_{timestamp}.json"
            else:
                filename = f"language_pack_{words_per_concept}w_{grammar}_{tier_short}_{timestamp}.json"

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
            self.log_status(f"Tier: {tier['name']}")

            # Estimate authentication rounds
            words_per_exchange = 8
            estimated_rounds = stats['total_words'] // words_per_exchange
            self.log_status(f"Estimated Auth Rounds: ~{estimated_rounds:,}")
            self.log_status("=" * 60)

            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Language pack generated successfully!\n\n"
                f"Tier: {tier['name']}\n"
                f"File: {filename}\n"
                f"Size: {file_size_mb:.2f} MB\n"
                f"Words: {stats['total_words']:,}"
            ))

        except Exception as e:
            self.log_status(f"❌ ERROR: {e}")
            import traceback
            self.log_status(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", f"Generation failed:\n{e}"))

        finally:
            # Re-enable UI
            self.root.after(0, self._generation_complete)

    def update_extraction_progress(self, current, total, message):
        """Update progress during WordNet extraction."""
        self.log_status(message)

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
