#!/usr/bin/env python3
"""
GREMLIN Admin Console - Unified God-Mode Interface
Beautiful Textual UI showing complete system view.
"""

import argparse
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Label, ProgressBar, Log
from textual.reactive import reactive
from textual import on
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from engine import GremlinEngine


class ModelVisual(Static):
    """Visual representation of a GREMLIN model (Pac-Man style)."""

    def __init__(self, name: str, side: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = name
        self.side = side  # 'left' or 'right'
        self.pack_color = "green"

    def render(self) -> Text:
        """Render the Pac-Man model."""
        if self.side == 'left':
            # Left-facing Pac-Man
            art = f"""
   ┌──────┐
   │ ◖◗   │
   │ {self.pack_color[0].upper()}    │
   └──────┘
  {self.model_name}
            """
        else:
            # Right-facing Pac-Man
            art = f"""
   ┌──────┐
   │   ◖◗ │
   │    {self.pack_color[0].upper()} │
   └──────┘
  {self.model_name}
            """

        return Text(art, style=self.pack_color)

    def set_pack_color(self, color: str):
        """Change the pack/lips color."""
        self.pack_color = color
        self.refresh()


class CommunicationLine(Static):
    """Visual representation of communication line with MITM."""

    direction = reactive("→")
    intercepted = reactive("")

    def render(self) -> Text:
        """Render the communication line."""
        if self.direction == "→":
            arrow = "━━━━━━━━━━━━━►"
        elif self.direction == "←":
            arrow = "◄━━━━━━━━━━━━━"
        else:
            arrow = "━━━━━━━━━━━━━━"

        line = f"""
        {arrow}
          [MITM 👁️]
     {self.intercepted[:20]}...
        """

        return Text(line, style="yellow")

    def show_packet(self, synthetic_text: str, direction: str):
        """Animate a packet transmission."""
        self.direction = direction
        self.intercepted = synthetic_text
        self.refresh()


class StatsPanel(Static):
    """Live statistics dashboard."""

    def __init__(self, engine: GremlinEngine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def render(self) -> Panel:
        """Render stats panel."""
        stats = self.engine.get_stats()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Language ID", stats['language_id'])
        table.add_row("Total Words", f"{stats['total_words']:,}")
        table.add_row("Words Used", f"{stats['words_used']:,}")
        table.add_row("Remaining", f"{stats['words_remaining']:,}")
        table.add_row("Usage", f"{stats['usage_percent']:.3f}%")
        table.add_row("Packets", str(stats['packet_count']))
        table.add_row("Attacks", str(stats['attack_count']))
        table.add_row("Est. Rounds", f"{stats['estimated_rounds']:,}")

        return Panel(table, title="📊 Live Stats", border_style="cyan")


class MessageLog(ScrollableContainer):
    """Scrollable message log."""

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.log_widget = Log()

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.title}[/bold]")
        yield self.log_widget

    def add_message(self, timestamp: str, text: str, style: str = "white"):
        """Add a message to the log."""
        self.log_widget.write(f"[{timestamp}] {text}", style=style)


class AdminConsole(App):
    """GREMLIN Admin Console Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
    }

    #models-container {
        height: 10;
        background: $surface-darken-1;
    }

    #client-model {
        width: 1fr;
        text-align: center;
    }

    #comm-line {
        width: 2fr;
        text-align: center;
    }

    #server-model {
        width: 1fr;
        text-align: center;
    }

    #logs-container {
        height: 1fr;
    }

    .log-panel {
        width: 1fr;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    #controls {
        dock: bottom;
        height: 5;
        background: $surface-darken-1;
    }

    #stats {
        width: 30;
        dock: right;
    }

    Button {
        margin: 0 1;
    }

    .success {
        background: $success;
    }

    .warning {
        background: $warning;
    }

    .error {
        background: $error;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "send_auth", "Send Auth"),
        ("x", "attack", "Attack Mode"),
        ("r", "rotate", "Rotate Language"),
    ]

    def __init__(self, engine: GremlinEngine):
        super().__init__()
        self.engine = engine
        self.engine.on_message = self.handle_new_message
        self.engine.on_stats_update = self.refresh_stats

    def compose(self) -> ComposeResult:
        """Create UI layout."""
        yield Header()

        # Title
        yield Static(
            "[bold cyan]GREMLIN ADMIN CONTROL CENTER[/bold cyan]",
            id="header"
        )

        # Main container
        with Container():
            # Stats panel (right side)
            with Vertical(id="stats"):
                yield StatsPanel(self.engine)

            # Models visualization
            with Horizontal(id="models-container"):
                yield ModelVisual("CLIENT 🟢", "left", id="client-model")
                yield CommunicationLine(id="comm-line")
                yield ModelVisual("SERVER 🟢", "right", id="server-model")

            # Message logs (split view)
            with Horizontal(id="logs-container"):
                yield MessageLog("CLIENT OUTPUT", classes="log-panel", id="client-log")
                yield MessageLog("MITM VIEW", classes="log-panel", id="mitm-log")
                yield MessageLog("SERVER OUTPUT", classes="log-panel", id="server-log")

            # Controls
            with Horizontal(id="controls"):
                yield Button("Send Auth", variant="success", id="btn-auth")
                yield Button("Attack Mode", variant="warning", id="btn-attack")
                yield Button("Rotate Language", variant="primary", id="btn-rotate")
                yield Button("New Pack", variant="primary", id="btn-newpack")
                yield Button("Quit", variant="error", id="btn-quit")

        yield Footer()

    def handle_new_message(self, message):
        """Handle new message from engine."""
        # Update comm line
        comm_line = self.query_one("#comm-line", CommunicationLine)
        direction = "→" if message.direction == "client->server" else "←"
        comm_line.show_packet(message.synthetic, direction)

        # Add to appropriate logs
        if message.sender == "client":
            client_log = self.query_one("#client-log", MessageLog)
            client_log.add_message(message.timestamp, f"[EN] {message.english}", "green")
            client_log.add_message(message.timestamp, f"[SYN] {message.synthetic}", "yellow")

            # MITM sees gibberish
            mitm_log = self.query_one("#mitm-log", MessageLog)
            mitm_log.add_message(message.timestamp, f"[INTERCEPTED] {message.synthetic}", "red")

        else:  # server
            server_log = self.query_one("#server-log", MessageLog)
            server_log.add_message(message.timestamp, f"[SYN] {message.synthetic}", "yellow")
            server_log.add_message(message.timestamp, f"[EN] {message.english}", "green")

            # MITM sees gibberish
            mitm_log = self.query_one("#mitm-log", MessageLog)
            mitm_log.add_message(message.timestamp, f"[INTERCEPTED] {message.synthetic}", "red")

        # Refresh stats
        self.refresh_stats()

    def refresh_stats(self):
        """Refresh statistics display."""
        stats_panel = self.query_one(StatsPanel)
        stats_panel.refresh()

    @on(Button.Pressed, "#btn-auth")
    def action_send_auth(self):
        """Send authentication message."""
        self.engine.send_authentication()

    @on(Button.Pressed, "#btn-attack")
    def action_attack(self):
        """Simulate attack."""
        gibberish = self.engine.simulate_attack()
        mitm_log = self.query_one("#mitm-log", MessageLog)
        mitm_log.add_message("--:--:--", f"[ATTACK ATTEMPT] {gibberish}", "red bold")
        mitm_log.add_message("--:--:--", "⚠️  Cannot decrypt without language pack!", "red")

    @on(Button.Pressed, "#btn-rotate")
    def action_rotate(self):
        """Rotate language (placeholder for now)."""
        mitm_log = self.query_one("#mitm-log", MessageLog)
        mitm_log.add_message("--:--:--", "🔄 Language rotation would happen here", "cyan")

        # Change model colors
        client_model = self.query_one("#client-model", ModelVisual)
        server_model = self.query_one("#server-model", ModelVisual)

        import random
        colors = ["green", "blue", "magenta", "cyan", "yellow"]
        new_color = random.choice(colors)

        client_model.set_pack_color(new_color)
        server_model.set_pack_color(new_color)

    @on(Button.Pressed, "#btn-newpack")
    def action_new_pack(self):
        """Generate new pack (placeholder)."""
        mitm_log = self.query_one("#mitm-log", MessageLog)
        mitm_log.add_message("--:--:--", "📦 New pack generation would happen here", "cyan")

    @on(Button.Pressed, "#btn-quit")
    def action_quit(self):
        """Quit application."""
        self.exit()


def main():
    parser = argparse.ArgumentParser(description="GREMLIN Admin Console")
    parser.add_argument('--pack', '-p', type=Path, required=True, help="Language pack file")
    parser.add_argument('--mode', choices=['demo', 'network'], default='demo', help="Mode")

    args = parser.parse_args()

    # Initialize engine
    engine = GremlinEngine(args.pack, mode=args.mode)

    # Run app
    app = AdminConsole(engine)
    app.run()


if __name__ == '__main__':
    main()
