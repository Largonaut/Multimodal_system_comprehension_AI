#!/usr/bin/env python3
"""
GREMLIN Language Pack Viewer
Browse and search generated language packs with full concept exploration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import json
import sys
import traceback


class LanguagePackViewer:
    """Comprehensive viewer for GREMLIN language packs."""

    def __init__(self, root):
        self.root = root
        self.root.title("GREMLIN Language Pack Viewer")
        self.root.geometry("1200x800")

        self.pack = None
        self.pack_path = None
        self.concepts_data = {}  # concept_id -> {definition, lemmas, pool}

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets."""

        # ============================================================
        # TOP BAR - File loading
        # ============================================================
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Language Pack:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.pack_label = ttk.Label(top_frame, text="No pack loaded", foreground='gray')
        self.pack_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(top_frame, text="📁 Load Pack...", command=self.load_pack).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="🔄 Reload", command=self.reload_pack).pack(side=tk.LEFT, padx=5)

        # ============================================================
        # MAIN LAYOUT - 3 panels
        # ============================================================
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL - Pack info + search
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)

        # Pack info
        info_label = ttk.Label(left_frame, text="📊 Pack Information", font=('Arial', 11, 'bold'))
        info_label.pack(pady=(0, 10))

        self.info_text = tk.Text(
            left_frame,
            height=8,
            bg='#f0f0f0',
            fg='#000000',
            font=('Courier', 9),
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.X, pady=(0, 15))

        # Search
        search_label = ttk.Label(left_frame, text="🔍 Search Concepts", font=('Arial', 11, 'bold'))
        search_label.pack(pady=(0, 5))

        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_concepts())

        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Arial', 10))
        search_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(
            search_frame,
            text="Search by English word, concept ID, or definition",
            font=('Arial', 8, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W)

        # Concept list
        list_label = ttk.Label(left_frame, text="📋 Concepts", font=('Arial', 11, 'bold'))
        list_label.pack(pady=(10, 5))

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.concept_listbox = tk.Listbox(
            list_frame,
            bg='#ffffff',
            fg='#000000',
            font=('Courier', 9),
            yscrollcommand=list_scroll.set,
            selectmode=tk.SINGLE
        )
        self.concept_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.concept_listbox.yview)

        self.concept_listbox.bind('<<ListboxSelect>>', self.on_concept_selected)

        # MIDDLE PANEL - Concept details
        middle_frame = ttk.Frame(main_paned)
        main_paned.add(middle_frame, weight=2)

        detail_label = ttk.Label(middle_frame, text="📖 Concept Details", font=('Arial', 11, 'bold'))
        detail_label.pack(pady=(0, 10))

        self.detail_text = scrolledtext.ScrolledText(
            middle_frame,
            bg='#f9f9f9',
            fg='#000000',
            font=('Courier', 10),
            wrap=tk.WORD
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # RIGHT PANEL - Synthetic words
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        words_label = ttk.Label(right_frame, text="🌍 Synthetic Words", font=('Arial', 11, 'bold'))
        words_label.pack(pady=(0, 10))

        # Filter controls
        filter_frame = ttk.Frame(right_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        self.word_filter = tk.StringVar(value='all')
        ttk.Radiobutton(filter_frame, text="All", variable=self.word_filter, value='all', command=self.refresh_word_list).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Unused", variable=self.word_filter, value='unused', command=self.refresh_word_list).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Used", variable=self.word_filter, value='used', command=self.refresh_word_list).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Use-Last", variable=self.word_filter, value='use_last', command=self.refresh_word_list).pack(side=tk.LEFT, padx=5)

        self.word_count_label = ttk.Label(filter_frame, text="", foreground='blue')
        self.word_count_label.pack(side=tk.RIGHT, padx=10)

        self.words_text = scrolledtext.ScrolledText(
            right_frame,
            bg='#ffffff',
            fg='#000000',
            font=('Courier', 10),
            wrap=tk.WORD
        )
        self.words_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Ready. Load a language pack to begin.", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

        # Initial state
        self.update_info_display()

    def load_pack(self):
        """Load a language pack file."""
        pack_path = filedialog.askopenfilename(
            title="Select Language Pack",
            initialdir="language_packs",
            filetypes=[("Language Packs", "*.json"), ("All Files", "*.*")]
        )

        if not pack_path:
            return

        self.pack_path = Path(pack_path)
        self.status_label.config(text=f"Loading {self.pack_path.name}...")
        self.root.update()

        try:
            # Load pack JSON
            with open(self.pack_path, 'r', encoding='utf-8') as f:
                pack_data = json.load(f)

            self.pack = pack_data
            self.concepts_data = {}

            # Check if WordNet is available
            wordnet_available = False
            try:
                from nltk.corpus import wordnet as wn
                # Test if we can access WordNet
                test = list(wn.all_synsets())[:1]
                wordnet_available = True
                self.status_label.config(text="WordNet available - loading definitions...")
            except Exception as e:
                self.status_label.config(text=f"Warning: WordNet not available - definitions may be limited")
                print(f"WordNet not available: {e}")

            self.root.update()

            # Parse concepts
            for concept_id, pool_data in pack_data['word_pools'].items():
                # Try to extract lemmas from WordNet concepts
                lemmas = []
                definition = ""

                if concept_id.startswith('wordnet_'):
                    # WordNet concept - parse synset name
                    # Format: wordnet_{word}_{pos}_{num}
                    parts = concept_id.replace('wordnet_', '').split('_')
                    if len(parts) >= 3:
                        word = parts[0]
                        lemmas.append(word)

                        # Try to get full synset data from WordNet
                        try:
                            from nltk.corpus import wordnet as wn
                            synset_name = f"{parts[0]}.{parts[1]}.{parts[2]}"
                            synset = wn.synset(synset_name)
                            definition = synset.definition()
                            lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
                        except Exception as e:
                            # Fallback if WordNet lookup fails
                            definition = f"[WordNet lookup failed: {parts[0]}]"
                            lemmas = [parts[0]]
                            print(f"Warning: Could not load WordNet definition for {concept_id}: {e}")
                    else:
                        definition = concept_id.replace('_', ' ')
                        lemmas = [concept_id]
                else:
                    # Base concept - try to get from concept dictionary
                    try:
                        from core.concepts import ConceptDictionary
                        cd = ConceptDictionary()
                        concept_obj = cd.get_concept(concept_id)
                        if concept_obj:
                            definition = concept_obj.description
                            lemmas = [concept_id.replace('_', ' ')]
                        else:
                            definition = concept_id.replace('_', ' ')
                            lemmas = [concept_id]
                    except:
                        definition = concept_id.replace('_', ' ')
                        lemmas = [concept_id]

                # Parse word pool - convert from {word: status} dict to separate lists
                words_dict = pool_data.get('words', {})
                pool_lists = {
                    'unused': [word for word, status in words_dict.items() if status == 'unused'],
                    'used': [word for word, status in words_dict.items() if status == 'used'],
                    'use_last': [word for word, status in words_dict.items() if status == 'use_last']
                }

                self.concepts_data[concept_id] = {
                    'definition': definition,
                    'lemmas': lemmas,
                    'pool': pool_lists
                }

            # Update UI
            self.pack_label.config(text=self.pack_path.name, foreground='green')
            self.update_info_display()
            self.populate_concept_list()
            self.status_label.config(text=f"Loaded {len(self.concepts_data)} concepts from {self.pack_path.name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load language pack:\n{e}")
            self.status_label.config(text="Error loading pack")

    def reload_pack(self):
        """Reload the current pack."""
        if self.pack_path:
            self.load_pack_from_path(self.pack_path)

    def load_pack_from_path(self, path):
        """Load pack from specific path."""
        self.pack_path = path
        self.load_pack()

    def update_info_display(self):
        """Update the pack information display."""
        self.info_text.delete('1.0', tk.END)

        if not self.pack:
            self.info_text.insert(tk.END, "No pack loaded.\n\nUse 'Load Pack' to open a language pack file.")
            return

        # Display pack stats
        total_unused = sum(len(c['pool']['unused']) for c in self.concepts_data.values())
        total_used = sum(len(c['pool']['used']) for c in self.concepts_data.values())
        total_use_last = sum(len(c['pool']['use_last']) for c in self.concepts_data.values())

        grammar_data = self.pack.get('grammar', {})
        if isinstance(grammar_data, dict):
            grammar = grammar_data.get('word_order', 'Unknown')
        else:
            grammar = 'Unknown'

        stats = {
            'Language ID': self.pack.get('language_id', 'Unknown')[:16] + '...',
            'Grammar': grammar,
            'Total Concepts': f"{len(self.concepts_data):,}",
            'Total Words': f"{total_unused + total_used + total_use_last:,}",
            'Unused Words': f"{total_unused:,}",
            'Used Words': f"{total_used:,}",
        }

        for key, value in stats.items():
            self.info_text.insert(tk.END, f"{key}:\n  {value}\n\n")

    def populate_concept_list(self):
        """Populate the concept listbox."""
        self.concept_listbox.delete(0, tk.END)

        if not self.concepts_data:
            return

        # Sort concepts
        sorted_concepts = sorted(self.concepts_data.keys())

        for concept_id in sorted_concepts:
            data = self.concepts_data[concept_id]
            pool = data['pool']
            unused = len(pool['unused'])
            total = unused + len(pool['used']) + len(pool['use_last'])

            # Format: "concept_id | lemma | unused/total"
            lemma = data['lemmas'][0] if data['lemmas'] else ''
            display = f"{concept_id[:30]:30s} │ {lemma[:20]:20s} │ {unused:6,}/{total:6,}"
            self.concept_listbox.insert(tk.END, display)

    def filter_concepts(self):
        """Filter concepts based on search query."""
        if not self.concepts_data:
            return

        query = self.search_var.get().lower()
        self.concept_listbox.delete(0, tk.END)

        sorted_concepts = sorted(self.concepts_data.keys())

        for concept_id in sorted_concepts:
            data = self.concepts_data[concept_id]

            # Search in concept ID, lemmas, and definition
            searchable = [
                concept_id.lower(),
                data['definition'].lower(),
                *[lemma.lower() for lemma in data['lemmas']]
            ]

            if not query or any(query in s for s in searchable):
                pool = data['pool']
                unused = len(pool['unused'])
                total = unused + len(pool['used']) + len(pool['use_last'])

                lemma = data['lemmas'][0] if data['lemmas'] else ''
                display = f"{concept_id[:30]:30s} │ {lemma[:20]:20s} │ {unused:6,}/{total:6,}"
                self.concept_listbox.insert(tk.END, display)

    def on_concept_selected(self, event):
        """Handle concept selection."""
        selection = self.concept_listbox.curselection()
        if not selection:
            return

        # Parse concept ID from display
        item = self.concept_listbox.get(selection[0])
        concept_id = item.split('│')[0].strip()

        self.show_concept_details(concept_id)

    def show_concept_details(self, concept_id):
        """Show detailed information about a concept."""
        if concept_id not in self.concepts_data:
            return

        data = self.concepts_data[concept_id]
        pool = data['pool']

        # Clear detail text
        self.detail_text.delete('1.0', tk.END)

        # Header
        self.detail_text.insert(tk.END, "═" * 60 + "\n")
        self.detail_text.insert(tk.END, f"CONCEPT: {concept_id}\n")
        self.detail_text.insert(tk.END, "═" * 60 + "\n\n")

        # Definition - prominently displayed
        self.detail_text.insert(tk.END, "📖 DEFINITION:\n", 'bold')
        self.detail_text.insert(tk.END, "─" * 60 + "\n")
        if data['definition']:
            self.detail_text.insert(tk.END, f"{data['definition']}\n")
        else:
            self.detail_text.insert(tk.END, "[No definition available - WordNet may not be installed]\n")
        self.detail_text.insert(tk.END, "─" * 60 + "\n\n")

        # Configure bold tag
        self.detail_text.tag_config('bold', font=('Courier', 10, 'bold'))

        # Lemmas (English words/synonyms)
        self.detail_text.insert(tk.END, "English Words (Lemmas):\n")
        if data['lemmas']:
            for lemma in data['lemmas']:
                self.detail_text.insert(tk.END, f"  • {lemma}\n")
        else:
            self.detail_text.insert(tk.END, "  (No lemmas available)\n")
        self.detail_text.insert(tk.END, "\n")

        # Pool statistics
        self.detail_text.insert(tk.END, "Word Pool Statistics:\n")
        self.detail_text.insert(tk.END, f"  Unused:   {len(pool['unused']):,}\n")
        self.detail_text.insert(tk.END, f"  Used:     {len(pool['used']):,}\n")
        self.detail_text.insert(tk.END, f"  Use-Last: {len(pool['use_last']):,}\n")
        self.detail_text.insert(tk.END, f"  Total:    {len(pool['unused']) + len(pool['used']) + len(pool['use_last']):,}\n\n")

        # Note about synthetic words
        self.detail_text.insert(tk.END, "─" * 60 + "\n")
        self.detail_text.insert(tk.END, "View synthetic words in the right panel →\n")
        self.detail_text.insert(tk.END, "Filter by: All, Unused, Used, or Use-Last\n")

        # Refresh word list
        self.current_concept_id = concept_id
        self.refresh_word_list()

    def refresh_word_list(self):
        """Refresh the synthetic words list based on filter."""
        if not hasattr(self, 'current_concept_id'):
            return

        concept_id = self.current_concept_id
        if concept_id not in self.concepts_data:
            return

        data = self.concepts_data[concept_id]
        pool = data['pool']
        filter_type = self.word_filter.get()

        # Clear words text
        self.words_text.delete('1.0', tk.END)

        # Get words based on filter
        if filter_type == 'all':
            words = pool['unused'] + pool['used'] + pool['use_last']
            status = "All Words"
        elif filter_type == 'unused':
            words = pool['unused']
            status = "Unused Words"
        elif filter_type == 'used':
            words = pool['used']
            status = "Used Words"
        elif filter_type == 'use_last':
            words = pool['use_last']
            status = "Use-Last Words (DDoS Pool)"
        else:
            words = []
            status = ""

        # Update count label
        self.word_count_label.config(text=f"{len(words):,} {status}")

        # Display words
        if not words:
            self.words_text.insert(tk.END, f"No {status.lower()} available for this concept.\n")
            return

        # Show header
        self.words_text.insert(tk.END, f"{status} for '{data['lemmas'][0] if data['lemmas'] else concept_id}':\n")
        self.words_text.insert(tk.END, "─" * 60 + "\n\n")

        # Lazy loading - only show first 1000 words at a time
        MAX_DISPLAY = 1000
        display_words = words[:MAX_DISPLAY]

        # Show words in columns
        words_per_line = 4
        for i, word in enumerate(display_words, 1):
            self.words_text.insert(tk.END, f"{word:20s}")
            if i % words_per_line == 0:
                self.words_text.insert(tk.END, "\n")

        if len(display_words) % words_per_line != 0:
            self.words_text.insert(tk.END, "\n")

        self.words_text.insert(tk.END, f"\n─" * 60 + "\n")

        total_in_pool = len(pool['unused']) + len(pool['used']) + len(pool['use_last'])

        if len(words) > MAX_DISPLAY:
            self.words_text.insert(tk.END, f"Showing first {MAX_DISPLAY:,} of {len(words):,} {status.lower()}\n")
            self.words_text.insert(tk.END, f"(Total in concept: {total_in_pool:,} words)\n\n")
            self.words_text.insert(tk.END, "💡 Tip: Use filters to narrow down the list\n")
        else:
            self.words_text.insert(tk.END, f"Showing all {len(words):,} {status.lower()}\n")
            self.words_text.insert(tk.END, f"(Total in concept: {total_in_pool:,} words)")


def main():
    """Run the language pack viewer."""
    try:
        # Print startup diagnostics
        print("=" * 60)
        print("GREMLIN Language Pack Viewer - Startup Diagnostics")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Tkinter version: {tk.TkVersion}")
        print()

        # Check for WordNet availability
        try:
            from nltk.corpus import wordnet as wn
            print("✓ WordNet available")
            print(f"  WordNet synsets: {len(list(wn.all_synsets())):,}")
        except ImportError:
            print("✗ WordNet NOT available (definitions will show fallback messages)")
        except Exception as e:
            print(f"✗ WordNet check failed: {e}")

        print()
        print("Starting GUI...")
        print("=" * 60)
        print()

        root = tk.Tk()

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        app = LanguagePackViewer(root)
        root.mainloop()

        print("\nViewer closed successfully.")

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\nThis usually means a required module is missing.")
        print("Please ensure tkinter is installed with your Python distribution.")
        print(f"\nPython path: {sys.executable}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("\nFull error trace:")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == '__main__':
    main()
