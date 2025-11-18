#!/usr/bin/env python3
"""
GREMLIN Admin Console - Tkinter GUI
Real demonstration interface with language viewer.
"""

import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
import random
import os
import sys
import subprocess
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import GremlinEngine


class LanguageViewer(tk.Frame):
    """Widget for viewing language pack contents."""

    def __init__(self, parent, engine):
        super().__init__(parent)
        self.engine = engine
        self.setup_ui()
        self.refresh_data()

    def setup_ui(self):
        """Build the language viewer UI."""
        # Header
        header = tk.Label(
            self,
            text="📖 Language Pack Viewer",
            font=('Arial', 14, 'bold'),
            bg='#2d2d2d',
            fg='white'
        )
        header.pack(fill=tk.X, pady=10)

        # Pack info
        info_frame = tk.Frame(self, bg='#2d2d2d')
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_text = tk.Text(
            info_frame,
            height=6,
            bg='#1e1e1e',
            fg='#00ffff',
            font=('Courier', 10),
            relief=tk.FLAT
        )
        self.info_text.pack(fill=tk.X)

        # Search/filter
        filter_frame = tk.Frame(self, bg='#2d2d2d')
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(filter_frame, text="Filter Concept:", bg='#2d2d2d', fg='white').pack(side=tk.LEFT, padx=5)
        self.filter_entry = tk.Entry(filter_frame, width=20)
        self.filter_entry.pack(side=tk.LEFT, padx=5)
        self.filter_entry.bind('<KeyRelease>', lambda e: self.refresh_data())

        ttk.Button(filter_frame, text="Refresh", command=self.refresh_data).pack(side=tk.LEFT, padx=5)

        # Concept list
        list_frame = tk.Frame(self, bg='#2d2d2d')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Scrollable list
        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.concept_list = tk.Listbox(
            list_frame,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Courier', 9),
            yscrollcommand=scroll.set,
            selectmode=tk.SINGLE
        )
        self.concept_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.concept_list.yview)

        self.concept_list.bind('<<ListboxSelect>>', self.show_concept_details)

        # Details panel
        details_frame = tk.Frame(self, bg='#2d2d2d')
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Label(details_frame, text="Selected Concept Details:", bg='#2d2d2d', fg='white').pack(anchor=tk.W)

        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffff00',
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def refresh_data(self):
        """Refresh the language pack data."""
        # Update pack info
        self.info_text.delete('1.0', tk.END)
        stats = self.engine.pack.get_stats()
        self.info_text.insert(tk.END, f"Language ID: {stats['language_id']}\n")
        self.info_text.insert(tk.END, f"Total Concepts: {stats['total_concepts']}\n")
        self.info_text.insert(tk.END, f"Total Words: {stats['total_words']:,}\n")
        self.info_text.insert(tk.END, f"Words Remaining: {stats['words_remaining']:,}\n")
        self.info_text.insert(tk.END, f"Used Words: {stats['total_used']:,}\n")
        self.info_text.insert(tk.END, f"Usage: {stats['usage_percentage']:.1f}%\n")
        self.info_text.insert(tk.END, f"Grammar: {self.engine.pack.grammar.word_order}\n")

        # Update concept list
        self.concept_list.delete(0, tk.END)
        filter_text = self.filter_entry.get().lower()

        for concept_id in sorted(self.engine.pack.word_pools.keys()):
            if filter_text and filter_text not in concept_id.lower():
                continue

            pool = self.engine.pack.word_pools[concept_id]
            unused = len(pool.unused)
            used = len(pool.used)
            total = unused + used + len(pool.use_last)

            self.concept_list.insert(tk.END, f"{concept_id:30s} │ {unused:6,}/{total:6,} unused")

    def show_concept_details(self, event):
        """Show details for selected concept."""
        selection = self.concept_list.curselection()
        if not selection:
            return

        # Parse concept ID from list item
        item = self.concept_list.get(selection[0])
        concept_id = item.split('│')[0].strip()

        # Get concept info
        concept = self.engine.concept_dict.get_concept(concept_id)
        pool = self.engine.pack.word_pools.get(concept_id)

        if not concept or not pool:
            return

        # Show details
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert(tk.END, f"═══════════════════════════════════════\n")
        self.details_text.insert(tk.END, f"CONCEPT: {concept.id}\n")
        self.details_text.insert(tk.END, f"═══════════════════════════════════════\n\n")

        self.details_text.insert(tk.END, f"Category: {concept.category}\n")
        self.details_text.insert(tk.END, f"Description: {concept.description}\n\n")

        self.details_text.insert(tk.END, f"Word Pool Statistics:\n")
        self.details_text.insert(tk.END, f"  Unused:   {len(pool.unused):,}\n")
        self.details_text.insert(tk.END, f"  Used:     {len(pool.used):,}\n")
        self.details_text.insert(tk.END, f"  Use-Last: {len(pool.use_last):,}\n")
        self.details_text.insert(tk.END, f"  Total:    {len(pool.unused) + len(pool.used) + len(pool.use_last):,}\n\n")

        # Show sample words
        self.details_text.insert(tk.END, f"Sample Unused Words (first 20):\n")
        for i, word in enumerate(list(pool.unused)[:20], 1):
            self.details_text.insert(tk.END, f"  {i:2d}. {word}\n")

        if len(pool.unused) > 20:
            self.details_text.insert(tk.END, f"  ... and {len(pool.unused) - 20:,} more\n")


class GremlinAdminGUI:
    """GREMLIN Admin Console using Tkinter."""

    def __init__(self, engine: GremlinEngine):
        self.engine = engine
        self.engine.on_message = self.handle_new_message
        self.engine.on_stats_update = self.update_stats

        # Create main window
        self.root = tk.Tk()
        self.root.title("GREMLIN Admin Console - God Mode")
        self.root.geometry("1400x900")

        # UI Settings
        self.ui_scale = 1.0  # Default scale factor

        # Color scheme
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'client': '#00ff00',
            'server': '#00aaff',
            'mitm': '#ff4444',
            'panel': '#2d2d2d'
        }

        self.setup_ui()
        self.update_stats()

    def setup_ui(self):
        """Build the UI layout."""
        # Configure root
        self.root.configure(bg=self.colors['bg'])

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Language Pack...", command=self.load_language_pack)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Generate New Language Pack...", command=self.launch_generator)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="UI Scale...", command=self.open_scale_settings)

        # Header
        header = tk.Frame(self.root, bg='#0066cc', height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="🔒 GREMLIN ADMIN CONTROL CENTER 🔒",
            font=('Courier', 18, 'bold'),
            bg='#0066cc',
            fg='white'
        )
        title.pack(pady=15)

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Communication
        comm_tab = tk.Frame(notebook, bg=self.colors['bg'])
        notebook.add(comm_tab, text="💬 Communication")
        self.setup_comm_tab(comm_tab)

        # Tab 2: Language Viewer
        viewer_tab = tk.Frame(notebook, bg=self.colors['bg'])
        notebook.add(viewer_tab, text="📖 Language Viewer")
        self.language_viewer = LanguageViewer(viewer_tab, self.engine)
        self.language_viewer.pack(fill=tk.BOTH, expand=True)
        self.language_viewer.configure(bg=self.colors['bg'])

    def setup_comm_tab(self, parent):
        """Setup the communication tab."""
        # Top section: Models visualization
        models_frame = tk.Frame(parent, bg=self.colors['bg'])
        models_frame.pack(fill=tk.X, pady=(0, 10))

        # CLIENT model
        self.client_frame = tk.Frame(models_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        self.client_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        tk.Label(
            self.client_frame,
            text="CLIENT 🟢",
            font=('Courier', 14, 'bold'),
            bg=self.colors['panel'],
            fg=self.colors['client']
        ).pack(pady=5)

        self.client_visual = tk.Label(
            self.client_frame,
            text="  ┌──────┐\n  │ ◖◗   │\n  │ 🟩   │\n  └──────┘",
            font=('Courier', 12),
            bg=self.colors['panel'],
            fg=self.colors['client']
        )
        self.client_visual.pack(pady=10)

        # MITM
        mitm_frame = tk.Frame(models_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        mitm_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        tk.Label(
            mitm_frame,
            text="MITM 👁️",
            font=('Courier', 14, 'bold'),
            bg=self.colors['panel'],
            fg=self.colors['mitm']
        ).pack(pady=5)

        self.mitm_visual = tk.Label(
            mitm_frame,
            text="━━━━━━━━►\n[INTERCEPT]\n◄━━━━━━━━",
            font=('Courier', 12),
            bg=self.colors['panel'],
            fg='#ffaa00'
        )
        self.mitm_visual.pack(pady=10)

        # SERVER model
        self.server_frame = tk.Frame(models_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        self.server_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        tk.Label(
            self.server_frame,
            text="SERVER 🟢",
            font=('Courier', 14, 'bold'),
            bg=self.colors['panel'],
            fg=self.colors['server']
        ).pack(pady=5)

        self.server_visual = tk.Label(
            self.server_frame,
            text="  ┌──────┐\n  │   ◖◗ │\n  │   🟩 │\n  └──────┘",
            font=('Courier', 12),
            bg=self.colors['panel'],
            fg=self.colors['server']
        )
        self.server_visual.pack(pady=10)

        # Stats panel (right side)
        stats_frame = tk.Frame(models_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        stats_frame.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH)

        tk.Label(
            stats_frame,
            text="📊 STATS",
            font=('Courier', 12, 'bold'),
            bg=self.colors['panel'],
            fg='#00ffff'
        ).pack(pady=5)

        self.stats_text = tk.Text(
            stats_frame,
            width=25,
            height=10,
            bg=self.colors['panel'],
            fg='white',
            font=('Courier', 9),
            relief=tk.FLAT
        )
        self.stats_text.pack(padx=5, pady=5)

        # Middle section: Log panels
        logs_frame = tk.Frame(parent, bg=self.colors['bg'])
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # CLIENT log
        client_log_frame = tk.LabelFrame(
            logs_frame,
            text="CLIENT TERMINAL",
            bg=self.colors['panel'],
            fg=self.colors['client'],
            font=('Courier', 10, 'bold')
        )
        client_log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self.client_log = scrolledtext.ScrolledText(
            client_log_frame,
            width=30,
            bg='#001100',
            fg=self.colors['client'],
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.client_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # MITM log
        mitm_log_frame = tk.LabelFrame(
            logs_frame,
            text="MITM VIEW (GIBBERISH)",
            bg=self.colors['panel'],
            fg=self.colors['mitm'],
            font=('Courier', 10, 'bold')
        )
        mitm_log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self.mitm_log = scrolledtext.ScrolledText(
            mitm_log_frame,
            width=30,
            bg='#110000',
            fg=self.colors['mitm'],
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.mitm_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # SERVER log
        server_log_frame = tk.LabelFrame(
            logs_frame,
            text="SERVER TERMINAL",
            bg=self.colors['panel'],
            fg=self.colors['server'],
            font=('Courier', 10, 'bold')
        )
        server_log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self.server_log = scrolledtext.ScrolledText(
            server_log_frame,
            width=30,
            bg='#000011',
            fg=self.colors['server'],
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.server_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom section: Message input
        controls_frame = tk.Frame(parent, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        controls_frame.pack(fill=tk.X)

        tk.Label(
            controls_frame,
            text="💬 SEND MESSAGE",
            font=('Courier', 11, 'bold'),
            bg=self.colors['panel'],
            fg='#ffff00'
        ).pack(pady=5)

        # Message entry
        msg_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        msg_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(
            msg_frame,
            text="Message:",
            font=('Courier', 10, 'bold'),
            bg=self.colors['panel'],
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

        self.message_entry = tk.Entry(
            msg_frame,
            font=('Courier', 10),
            bg='#333333',
            fg='white',
            insertbackground='white'
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.message_entry.bind('<Return>', lambda e: self.send_as_client())

        ttk.Button(
            msg_frame,
            text="Send as CLIENT →",
            command=self.send_as_client
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            msg_frame,
            text="← Send as SERVER",
            command=self.send_as_server
        ).pack(side=tk.LEFT, padx=5)

        # Clear logs button
        clear_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        clear_frame.pack(pady=5)

        ttk.Button(
            clear_frame,
            text="🗑️ Clear All Logs",
            command=self.clear_logs
        ).pack(side=tk.LEFT, padx=5)

    def send_as_client(self):
        """Send message from client side."""
        message = self.message_entry.get().strip()
        if not message:
            return

        # Translate message to synthetic
        try:
            synthetic = self.engine.translator.translate_to_synthetic(message)
            print(f"\n[DEBUG] Translation:")
            print(f"  English:   {message}")
            print(f"  Synthetic: {synthetic[:100]}..." if len(synthetic) > 100 else f"  Synthetic: {synthetic}")
        except Exception as e:
            print(f"\n[ERROR] Translation failed: {e}")
            synthetic = "[TRANSLATION ERROR]"

        # Create message
        from engine import Message
        from datetime import datetime

        msg = Message(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            direction='client->server',
            english=message,
            synthetic=synthetic,
            sender='client'
        )

        self.handle_new_message(msg)

        # Clear entry
        self.message_entry.delete(0, tk.END)

        # Update stats
        self.engine.packet_count += 1
        self.update_stats()

        # Refresh language viewer
        self.language_viewer.refresh_data()

    def send_as_server(self):
        """Send message from server side."""
        message = self.message_entry.get().strip()
        if not message:
            return

        # Translate message to synthetic
        try:
            synthetic = self.engine.translator.translate_to_synthetic(message)
            print(f"\n[DEBUG] Translation:")
            print(f"  English:   {message}")
            print(f"  Synthetic: {synthetic[:100]}..." if len(synthetic) > 100 else f"  Synthetic: {synthetic}")
        except Exception as e:
            print(f"\n[ERROR] Translation failed: {e}")
            synthetic = "[TRANSLATION ERROR]"

        # Create message
        from engine import Message
        from datetime import datetime

        msg = Message(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            direction='server->client',
            english=message,
            synthetic=synthetic,
            sender='server'
        )

        self.handle_new_message(msg)

        # Clear entry
        self.message_entry.delete(0, tk.END)

        # Update stats
        self.engine.packet_count += 1
        self.update_stats()

        # Refresh language viewer
        self.language_viewer.refresh_data()

    def clear_logs(self):
        """Clear all log panels."""
        self.client_log.delete('1.0', tk.END)
        self.mitm_log.delete('1.0', tk.END)
        self.server_log.delete('1.0', tk.END)

    def launch_generator(self):
        """Launch the language pack generator GUI."""
        generator_path = Path(__file__).parent.parent / "language_pack_generator_gui.py"

        if not generator_path.exists():
            messagebox.showerror("Error", f"Generator not found:\n{generator_path}")
            return

        # Launch in separate process
        try:
            if sys.platform == 'win32':
                subprocess.Popen([sys.executable, str(generator_path)],
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, str(generator_path)])

            messagebox.showinfo(
                "Generator Launched",
                "Language pack generator opened in new window.\n\n"
                "After generating a pack, use:\n"
                "File > Load Language Pack... to load it."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch generator:\n{e}")

    def load_language_pack(self):
        """Show file picker to load a new language pack and restart."""
        # Get language_packs directory
        default_dir = Path(__file__).parent.parent / "language_packs"
        if not default_dir.exists():
            default_dir = Path.cwd()

        # Show file picker
        pack_path = filedialog.askopenfilename(
            title="Select Language Pack",
            initialdir=default_dir,
            filetypes=[("Language Packs", "*.json"), ("All Files", "*.*")]
        )

        if not pack_path:
            return  # User cancelled

        # Confirm restart
        response = messagebox.askyesno(
            "Load Language Pack",
            f"Loading a new language pack will restart the application.\n\n"
            f"Selected: {Path(pack_path).name}\n\n"
            f"Continue?"
        )

        if response:
            # Restart the app with new pack
            self.root.destroy()
            os.execv(sys.executable, [sys.executable] + sys.argv[:1] + ['--pack', pack_path, '--mode', self.engine.mode])

    def handle_new_message(self, message):
        """Handle new message from engine."""
        if message.sender == "client":
            # Client terminal - shows what client SENDS
            self.log_client(f"[{message.timestamp}] CLIENT SENT:\n", 'green')
            self.log_client(f"  [EN]  {message.english}\n", 'white')
            self.log_client(f"  [SYN] {message.synthetic}\n\n", 'yellow')

            # Server terminal - shows what server RECEIVES (with translation)
            self.log_server(f"[{message.timestamp}] SERVER RECEIVED:\n", 'cyan')
            self.log_server(f"  [SYN] {message.synthetic}\n", 'yellow')
            self.log_server(f"  [EN]  {message.english} (translated)\n\n", 'white')

            # MITM sees gibberish
            self.log_mitm(f"[{message.timestamp}] [INTERCEPTED]\n{message.synthetic}\n\n", 'red')

        else:  # server
            # Server terminal - shows what server SENDS
            self.log_server(f"[{message.timestamp}] SERVER SENT:\n", 'green')
            self.log_server(f"  [EN]  {message.english}\n", 'white')
            self.log_server(f"  [SYN] {message.synthetic}\n\n", 'yellow')

            # Client terminal - shows what client RECEIVES (with translation)
            self.log_client(f"[{message.timestamp}] CLIENT RECEIVED:\n", 'cyan')
            self.log_client(f"  [SYN] {message.synthetic}\n", 'yellow')
            self.log_client(f"  [EN]  {message.english} (translated)\n\n", 'white')

            # MITM sees response gibberish
            self.log_mitm(f"[{message.timestamp}] [INTERCEPTED]\n{message.synthetic}\n\n", 'red')

        # Update stats
        self.update_stats()

        # Animate MITM visual
        self.animate_mitm()

    def animate_mitm(self):
        """Briefly animate MITM interception."""
        self.mitm_visual.config(fg='#ff0000')
        self.root.after(200, lambda: self.mitm_visual.config(fg='#ffaa00'))

    def log_client(self, text, color='white'):
        """Log to client panel."""
        self.client_log.insert(tk.END, text)
        self.client_log.see(tk.END)

    def log_mitm(self, text, color='white'):
        """Log to MITM panel."""
        self.mitm_log.insert(tk.END, text)
        self.mitm_log.see(tk.END)

    def log_server(self, text, color='white'):
        """Log to server panel."""
        self.server_log.insert(tk.END, text)
        self.server_log.see(tk.END)

    def update_stats(self):
        """Update statistics display."""
        stats = self.engine.get_stats()

        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, f"Lang ID:\n{stats['language_id'][:8]}...\n\n")
        self.stats_text.insert(tk.END, f"Words: {stats['total_words']:,}\n")
        self.stats_text.insert(tk.END, f"Remaining: {stats.get('words_remaining', 0):,}\n")
        self.stats_text.insert(tk.END, f"Used: {stats.get('total_used', 0):,}\n")
        self.stats_text.insert(tk.END, f"Packets: {stats['packet_count']}\n")
        self.stats_text.insert(tk.END, f"Est. Rounds: {stats['estimated_rounds']:,}\n")

    def open_scale_settings(self):
        """Open UI scale settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("UI Scale Settings")
        dialog.geometry("400x250")
        dialog.configure(bg=self.colors['panel'])
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        # Title
        title_frame = tk.Frame(dialog, bg='#0066cc', height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="⚙️ UI Scale Settings",
            font=('Courier', 14, 'bold'),
            bg='#0066cc',
            fg='white'
        ).pack(pady=10)

        # Content frame
        content = tk.Frame(dialog, bg=self.colors['panel'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Current scale display
        scale_var = tk.DoubleVar(value=self.ui_scale)
        scale_label = tk.Label(
            content,
            text=f"Current Scale: {self.ui_scale:.1f}x",
            font=('Courier', 12, 'bold'),
            bg=self.colors['panel'],
            fg='#00ff00'
        )
        scale_label.pack(pady=(0, 20))

        # Description
        tk.Label(
            content,
            text="Adjust the UI scale (requires restart to take full effect)",
            font=('Arial', 9),
            bg=self.colors['panel'],
            fg='#aaaaaa',
            wraplength=350
        ).pack(pady=(0, 10))

        # Scale slider
        def update_scale_label(value):
            scale_label.config(text=f"Current Scale: {float(value):.1f}x")

        scale_slider = tk.Scale(
            content,
            from_=0.5,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=scale_var,
            command=update_scale_label,
            bg=self.colors['panel'],
            fg='white',
            highlightthickness=0,
            length=300,
            font=('Arial', 9)
        )
        scale_slider.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['panel'])
        button_frame.pack(pady=10)

        def apply_scale():
            new_scale = scale_var.get()
            old_scale = self.ui_scale
            self.ui_scale = new_scale

            # Apply window scaling
            base_width = 1400
            base_height = 900
            new_width = int(base_width * new_scale)
            new_height = int(base_height * new_scale)

            self.root.geometry(f"{new_width}x{new_height}")

            # Apply tkinter global scaling
            import tkinter.font as tkfont
            default_font = tkfont.nametofont("TkDefaultFont")
            text_font = tkfont.nametofont("TkTextFont")
            fixed_font = tkfont.nametofont("TkFixedFont")

            base_size = 9
            new_size = max(6, int(base_size * new_scale))

            default_font.configure(size=new_size)
            text_font.configure(size=new_size)
            fixed_font.configure(size=new_size)

            # Force update all widgets
            self.root.update_idletasks()

            messagebox.showinfo(
                "Scale Applied",
                f"UI scale set to {new_scale:.1f}x\n\n"
                f"Window: {new_width}x{new_height}\n"
                f"Font size: {new_size}pt"
            )
            dialog.destroy()

        ttk.Button(
            button_frame,
            text="Apply",
            command=apply_scale
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def main():
    try:
        # Print startup diagnostics
        print("=" * 60)
        print("GREMLIN Admin Console - Startup Diagnostics")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Tkinter version: {tk.TkVersion}")
        print()
        print("Starting admin console...")
        print("=" * 60)
        print()

        parser = argparse.ArgumentParser(description="GREMLIN Admin Console (Tkinter)")
        parser.add_argument('--pack', '-p', type=Path, required=False, help="Language pack file (optional, will show picker if not provided)")
        parser.add_argument('--mode', choices=['demo', 'network'], default='demo', help="Mode")

        args = parser.parse_args()

        # If no pack specified, show file picker
        pack_path = args.pack
        if not pack_path:
            # Create temporary Tk window for file picker
            root = tk.Tk()
            root.withdraw()  # Hide the main window

            default_dir = Path(__file__).parent.parent / "language_packs"
            if not default_dir.exists():
                default_dir = Path.cwd()

            pack_path = filedialog.askopenfilename(
                title="Select Language Pack",
                initialdir=default_dir,
                filetypes=[("Language Packs", "*.json"), ("All Files", "*.*")]
            )

            root.destroy()

            if not pack_path:
                print("No language pack selected. Exiting.")
                return

            pack_path = Path(pack_path)

        # Initialize engine with progress indicators
        print("\n📦 Loading language pack...")
        print(f"   File: {pack_path.name}")
        print(f"   Size: {pack_path.stat().st_size / (1024*1024):.1f} MB")
        print()

        import time
        start_time = time.time()

        # Show spinner during load
        import threading
        loading = True
        def spinner():
            chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            i = 0
            while loading:
                print(f'\r   {chars[i % len(chars)]} Loading...', end='', flush=True)
                time.sleep(0.1)
                i += 1
            print('\r   ✓ Loaded!     ')

        spinner_thread = threading.Thread(target=spinner, daemon=True)
        spinner_thread.start()

        engine = GremlinEngine(pack_path, mode=args.mode)

        loading = False
        spinner_thread.join(timeout=0.5)

        elapsed = time.time() - start_time
        print(f"   Time: {elapsed:.1f}s")
        print()

        # Run GUI
        app = GremlinAdminGUI(engine)
        app.run()

        print("\nAdmin console closed successfully.")

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\nThis usually means a required module is missing.")
        print("Please ensure all dependencies are installed.")
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
