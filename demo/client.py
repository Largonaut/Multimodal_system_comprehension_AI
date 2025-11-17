#!/usr/bin/env python3
"""
GREMLIN Client Demo (Laptop 1)
Authenticates with server using synthetic language.
"""

import argparse
import socket
import json
import random
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConceptDictionary, LanguagePack, GrammarEngine
from translation import GemmaTranslator


class GremlinClient:
    """Client that authenticates using synthetic language."""

    def __init__(
        self,
        language_pack_path: Path,
        server_host: str = "localhost",
        server_port: int = 9999,
        gemma_path: Optional[str] = None
    ):
        """
        Initialize GREMLIN client.

        Args:
            language_pack_path: Path to language pack
            server_host: Server hostname
            server_port: Server port
            gemma_path: Path to Gemma 2 model (optional)
        """
        self.console = Console()
        self.host = server_host
        self.port = server_port

        # Load language pack
        self.console.print(f"[cyan]Loading language pack from {language_pack_path}...[/cyan]")
        self.pack = LanguagePack.load(language_pack_path)

        # Load concept dictionary and grammar
        self.concept_dict = ConceptDictionary()
        self.grammar = GrammarEngine()

        # Initialize translator
        self.console.print("[cyan]Initializing translator...[/cyan]")
        self.translator = GemmaTranslator(
            self.pack,
            self.concept_dict,
            gemma_model_path=gemma_path,
            use_ai=bool(gemma_path)
        )

        self.history = []

    def authenticate(self, name: str = None, company: str = None, topic: str = None):
        """
        Send authentication message.

        Args:
            name: Name to use (random if None)
            company: Company to use (random if None)
            topic: Topic to use (random if None)

        Returns:
            Tuple of (english_msg, synthetic_msg, reverse_map)
        """
        # Pick random values if not provided
        if not name:
            names = self.concept_dict.get_concepts_by_category('variables_names')
            name_concept = random.choice(names)
            name = name_concept.term
            name_id = name_concept.id
        else:
            matches = self.concept_dict.search_by_term(name)
            name_id = matches[0].id if matches else None

        if not company:
            companies = self.concept_dict.get_concepts_by_category('variables_companies')
            company_concept = random.choice(companies)
            company = company_concept.term
            company_id = company_concept.id
        else:
            matches = self.concept_dict.search_by_term(company)
            company_id = matches[0].id if matches else None

        if not topic:
            topics = self.concept_dict.get_concepts_by_category('variables_topics')
            topic_concept = random.choice(topics)
            topic = topic_concept.term
            topic_id = topic_concept.id
        else:
            matches = self.concept_dict.search_by_term(topic)
            topic_id = matches[0].id if matches else None

        # Generate authentication message
        variables = {
            'name': name_id,
            'company': company_id,
            'topic': topic_id
        }

        english_msg = f"Checking in, this is {name} with {company} talking about {topic}"
        synthetic_msg = self.translator.translate_to_synthetic(
            english_msg,
            template_name='authentication_client',
            variables=variables
        )

        # Create reverse map for server
        reverse_map = self.translator.create_reverse_map(synthetic_msg)

        return english_msg, synthetic_msg, reverse_map

    def send_message(self, synthetic_msg: str, reverse_map: dict, timeout: float = 5.0):
        """
        Send message to server.

        Args:
            synthetic_msg: Synthetic language message
            reverse_map: Reverse mapping for server
            timeout: Socket timeout

        Returns:
            Server response (synthetic) or None
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect((self.host, self.port))

                # Send message + reverse map
                payload = {
                    'message': synthetic_msg,
                    'reverse_map': reverse_map
                }
                sock.sendall(json.dumps(payload).encode() + b'\n')

                # Receive response
                response = sock.recv(4096).decode()
                return json.loads(response)['message']

        except (socket.timeout, ConnectionRefusedError, socket.error) as e:
            self.console.print(f"[red]Connection error: {e}[/red]")
            return None

    def run_interactive(self):
        """Run interactive demo mode."""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]GREMLIN Client[/bold green]\n"
            f"[dim]Connected to: {self.host}:{self.port}[/dim]\n"
            f"[dim]Language: {self.pack.language_id[:16]}...[/dim]",
            box=box.DOUBLE
        ))

        while True:
            self.console.print("\n[yellow]━━━ Authentication Options ━━━[/yellow]")
            choice = Prompt.ask(
                "Choose",
                choices=["auto", "custom", "quit"],
                default="auto"
            )

            if choice == "quit":
                break

            if choice == "auto":
                name, company, topic = None, None, None
            else:
                name = Prompt.ask("Name (or press Enter for random)", default="")
                company = Prompt.ask("Company (or press Enter for random)", default="")
                topic = Prompt.ask("Topic (or press Enter for random)", default="")
                name = name if name else None
                company = company if company else None
                topic = topic if topic else None

            # Generate and send authentication
            eng_msg, syn_msg, rev_map = self.authenticate(name, company, topic)

            # Display client message
            table = Table(show_header=True, box=box.ROUNDED, border_style="green")
            table.add_column("CLIENT → SERVER", style="white")
            table.add_row(
                f"[dim]English:[/dim]\n{eng_msg}\n\n"
                f"[bold green]Synthetic:[/bold green]\n{syn_msg}"
            )
            self.console.print(table)

            # Send to server
            self.console.print("[dim]Sending to server...[/dim]")
            response = self.send_message(syn_msg, rev_map)

            if response:
                # Display server response
                table = Table(show_header=True, box=box.ROUNDED, border_style="blue")
                table.add_column("SERVER → CLIENT", style="white")

                # Translate response back to English
                eng_response = self.translator.translate_to_english(response, rev_map)

                table.add_row(
                    f"[bold blue]Synthetic:[/bold blue]\n{response}\n\n"
                    f"[dim]English:[/dim]\n{eng_response}"
                )
                self.console.print(table)

                # Add to history
                self.history.append({
                    'client_eng': eng_msg,
                    'client_syn': syn_msg,
                    'server_syn': response,
                    'server_eng': eng_response
                })

    def run_scripted(self, rounds: int = 5):
        """Run scripted demo (pre-programmed exchanges)."""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]GREMLIN Client - Scripted Demo[/bold green]\n"
            f"[dim]Rounds: {rounds}[/dim]",
            box=box.DOUBLE
        ))

        for i in range(1, rounds + 1):
            self.console.print(f"\n[yellow]━━━ Round {i}/{rounds} ━━━[/yellow]")

            # Generate authentication
            eng_msg, syn_msg, rev_map = self.authenticate()

            # Display
            table = Table(show_header=True, box=box.ROUNDED, border_style="green")
            table.add_column("CLIENT → SERVER", style="white")
            table.add_row(
                f"[dim]English:[/dim]\n{eng_msg}\n\n"
                f"[bold green]Synthetic:[/bold green]\n{syn_msg}"
            )
            self.console.print(table)

            # Simulate network transmission
            import time
            time.sleep(0.5)

        # Show stats
        stats = self.pack.get_stats()
        self.console.print(f"\n[cyan]Language pack usage: {stats['usage_percentage']:.3f}%[/cyan]")


def main():
    parser = argparse.ArgumentParser(description="GREMLIN Client Demo")
    parser.add_argument('--pack', '-p', type=Path, required=True, help="Language pack file")
    parser.add_argument('--host', default='localhost', help="Server host")
    parser.add_argument('--port', type=int, default=9999, help="Server port")
    parser.add_argument('--gemma', type=str, help="Path to Gemma 2 model")
    parser.add_argument('--mode', choices=['interactive', 'scripted'], default='interactive')
    parser.add_argument('--rounds', type=int, default=5, help="Rounds for scripted mode")

    args = parser.parse_args()

    client = GremlinClient(args.pack, args.host, args.port, args.gemma)

    if args.mode == 'interactive':
        client.run_interactive()
    else:
        client.run_scripted(args.rounds)


if __name__ == '__main__':
    from typing import Optional
    main()
