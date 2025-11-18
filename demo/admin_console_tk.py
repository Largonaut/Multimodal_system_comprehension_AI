#!/usr/bin/env python3
"""
GREMLIN Admin Console - Tkinter GUI
Clean, user-friendly god-mode interface.
"""

import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
import random
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import GremlinEngine


class GremlinAdminGUI:
    """GREMLIN Admin Console using Tkinter."""

    def __init__(self, engine: GremlinEngine):
        self.engine = engine
        self.engine.on_message = self.handle_new_message
        self.engine.on_stats_update = self.update_stats

        # Create main window
        self.root = tk.Tk()
        self.root.title("GREMLIN Admin Console - God Mode")
        self.root.geometry("1200x800")

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

        # Main content area
        content = tk.Frame(self.root, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top section: Models visualization
        models_frame = tk.Frame(content, bg=self.colors['bg'])
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
        logs_frame = tk.Frame(content, bg=self.colors['bg'])
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # CLIENT log
        client_log_frame = tk.LabelFrame(
            logs_frame,
            text="CLIENT OUTPUT",
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
            text="SERVER OUTPUT",
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

        # Bottom section: Controls
        controls_frame = tk.Frame(content, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        controls_frame.pack(fill=tk.X)

        tk.Label(
            controls_frame,
            text="ADMIN CONTROLS",
            font=('Courier', 11, 'bold'),
            bg=self.colors['panel'],
            fg='#ffff00'
        ).pack(pady=5)

        buttons_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        buttons_frame.pack(pady=5)

        # Buttons
        ttk.Button(
            buttons_frame,
            text="📤 Send Auth",
            command=self.send_auth
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="⚠️ Attack Mode",
            command=self.attack_mode
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="🔄 Rotate Language",
            command=self.rotate_language
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="🚀 Auto Send (5x)",
            command=self.auto_send
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="❌ Quit",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)

        # Chat mode
        chat_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        chat_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(
            chat_frame,
            text="💬 CHAT MODE:",
            font=('Courier', 10, 'bold'),
            bg=self.colors['panel'],
            fg='#00ff00'
        ).pack(side=tk.LEFT, padx=5)

        self.chat_entry = tk.Entry(
            chat_frame,
            font=('Courier', 10),
            bg='#333333',
            fg='white',
            insertbackground='white'
        )
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.chat_entry.bind('<Return>', lambda e: self.send_chat_message())

        ttk.Button(
            chat_frame,
            text="Send Message",
            command=self.send_chat_message
        ).pack(side=tk.LEFT, padx=5)

    def send_auth(self):
        """Send one authentication message."""
        self.engine.send_authentication()

    def attack_mode(self):
        """Simulate attack."""
        gibberish = self.engine.simulate_attack()
        self.log_mitm(f"\n⚠️ [ATTACK ATTEMPT] ⚠️\n{gibberish}\n⚠️ Cannot decrypt without language pack!\n", 'red')

    def rotate_language(self):
        """ACTUALLY rotate to a new language pack."""
        # Show progress
        self.log_mitm("\n🔄 Generating new language pack...\n", 'cyan')
        self.root.update()

        # Generate new pack
        from pathlib import Path
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / 'gremlin_temp'
        temp_dir.mkdir(exist_ok=True)

        new_pack_path = self.engine.generate_new_pack(
            words_per_concept=500,  # Smaller for speed
            output_path=temp_dir / f"temp_pack_{random.randint(1000, 9999)}.json"
        )

        self.log_mitm(f"✅ Generated: {new_pack_path.name}\n", 'green')
        self.root.update()

        # Load new pack
        self.log_mitm("🔄 Loading new pack...\n", 'cyan')
        self.root.update()

        self.engine.rotate_language(new_pack_path)

        # Clear all logs
        self.client_log.delete('1.0', tk.END)
        self.mitm_log.delete('1.0', tk.END)
        self.server_log.delete('1.0', tk.END)

        # Change visual colors
        colors = ['🟩', '🟦', '🟪', '🟨', '🟧']
        new_color = random.choice(colors)

        self.client_visual.config(text=f"  ┌──────┐\n  │ ◖◗   │\n  │ {new_color}   │\n  └──────┘")
        self.server_visual.config(text=f"  ┌──────┐\n  │   ◖◗ │\n  │   {new_color} │\n  └──────┘")

        self.log_mitm("\n✅ NEW LANGUAGE LOADED!\n", 'green')
        self.log_mitm("Old messages now worthless. Perfect forward secrecy!\n\n", 'cyan')

        # Update stats
        self.update_stats()

    def auto_send(self):
        """Send 5 auth messages automatically."""
        for _ in range(5):
            self.send_auth()
            self.root.update()

    def send_chat_message(self):
        """Send custom chat message."""
        message = self.chat_entry.get().strip()
        if not message:
            return

        # Translate message to synthetic
        synthetic = self.engine.translator.translate_to_synthetic(message)

        # Create fake message object
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

        # Generate server response (echo back)
        response_msg = Message(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            direction='server->client',
            english=f"Received: {message}",
            synthetic=self.engine.translator.translate_to_synthetic(f"Received: {message}"),
            sender='server'
        )

        self.handle_new_message(response_msg)

        # Clear entry
        self.chat_entry.delete(0, tk.END)

        # Update stats
        self.engine.packet_count += 2
        self.update_stats()

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
            # Client log
            self.log_client(f"[{message.timestamp}] [EN] {message.english}\n", 'green')
            self.log_client(f"[{message.timestamp}] [SYN] {message.synthetic}\n\n", 'yellow')

            # MITM sees gibberish
            self.log_mitm(f"[{message.timestamp}] [INTERCEPTED]\n{message.synthetic}\n\n", 'red')

        else:  # server
            # Server log
            self.log_server(f"[{message.timestamp}] [SYN] {message.synthetic}\n", 'yellow')
            self.log_server(f"[{message.timestamp}] [EN] {message.english}\n\n", 'cyan')

            # MITM sees response gibberish
            self.log_mitm(f"[{message.timestamp}] [INTERCEPTED]\n{message.synthetic}\n\n", 'red')

        # Update stats
        self.update_stats()

        # Animate MITM visual
        self.animate_mitm()

    def log_client(self, text, color='white'):
        """Add to client log."""
        self.client_log.insert(tk.END, text)
        self.client_log.see(tk.END)

    def log_mitm(self, text, color='white'):
        """Add to MITM log."""
        self.mitm_log.insert(tk.END, text)
        self.mitm_log.see(tk.END)

    def log_server(self, text, color='white'):
        """Add to server log."""
        self.server_log.insert(tk.END, text)
        self.server_log.see(tk.END)

    def animate_mitm(self):
        """Animate MITM visual briefly."""
        self.mitm_visual.config(fg='#ff0000')
        self.root.after(200, lambda: self.mitm_visual.config(fg='#ffaa00'))

    def update_stats(self):
        """Update stats panel."""
        stats = self.engine.get_stats()

        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, f"Language ID:\n{stats['language_id']}\n\n")
        self.stats_text.insert(tk.END, f"Total Words: {stats['total_words']:,}\n")
        self.stats_text.insert(tk.END, f"Words Used: {stats['words_used']:,}\n")
        self.stats_text.insert(tk.END, f"Remaining: {stats['words_remaining']:,}\n")
        self.stats_text.insert(tk.END, f"Usage: {stats['usage_percent']:.3f}%\n\n")
        self.stats_text.insert(tk.END, f"Packets: {stats['packet_count']}\n")
        self.stats_text.insert(tk.END, f"Attacks: {stats['attack_count']}\n")
        self.stats_text.insert(tk.END, f"Est. Rounds: {stats['estimated_rounds']:,}\n")

    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def main():
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

    # Initialize engine
    engine = GremlinEngine(pack_path, mode=args.mode)

    # Run GUI
    app = GremlinAdminGUI(engine)
    app.run()


if __name__ == '__main__':
    main()
