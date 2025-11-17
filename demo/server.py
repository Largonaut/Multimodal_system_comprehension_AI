#!/usr/bin/env python3
"""
GREMLIN Server Demo (Laptop 2)
Receives and authenticates synthetic language messages.
"""

import argparse
import socket
import json
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ConceptDictionary, LanguagePack, GrammarEngine
from translation import GemmaTranslator


class GremlinServer:
    """Server that authenticates clients using synthetic language."""

    def __init__(
        self,
        language_pack_path: Path,
        host: str = "0.0.0.0",
        port: int = 9999,
        gemma_path: Optional[str] = None
    ):
        """
        Initialize GREMLIN server.

        Args:
            language_pack_path: Path to language pack (must match client's)
            host: Host to bind to
            port: Port to listen on
            gemma_path: Path to Gemma 2 model (optional)
        """
        self.console = Console()
        self.host = host
        self.port = port

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

        self.request_count = 0

    def handle_client(self, conn, addr):
        """
        Handle a client connection.

        Args:
            conn: Socket connection
            addr: Client address
        """
        try:
            # Receive message
            data = conn.recv(4096).decode()
            payload = json.loads(data)

            synthetic_msg = payload['message']
            reverse_map = payload['reverse_map']

            self.request_count += 1

            # Translate to English
            english_msg = self.translator.translate_to_english(synthetic_msg, reverse_map)

            # Display received message
            self.console.print(f"\n[yellow]━━━ Request #{self.request_count} from {addr[0]} ━━━[/yellow]")

            table = Table(show_header=True, box=box.ROUNDED, border_style="green")
            table.add_column("CLIENT → SERVER", style="white")
            table.add_row(
                f"[bold green]Synthetic:[/bold green]\n{synthetic_msg}\n\n"
                f"[dim]English:[/dim]\n{english_msg}"
            )
            self.console.print(table)

            # Generate response
            # Extract variables from English message
            response_eng, response_syn = self._generate_response(english_msg, reverse_map)

            # Display response
            table = Table(show_header=True, box=box.ROUNDED, border_style="blue")
            table.add_column("SERVER → CLIENT", style="white")
            table.add_row(
                f"[dim]English:[/dim]\n{response_eng}\n\n"
                f"[bold blue]Synthetic:[/bold blue]\n{response_syn}"
            )
            self.console.print(table)

            # Send response
            response_payload = {'message': response_syn}
            conn.sendall(json.dumps(response_payload).encode())

        except Exception as e:
            self.console.print(f"[red]Error handling client: {e}[/red]")
        finally:
            conn.close()

    def _generate_response(self, english_msg: str, reverse_map: dict):
        """
        Generate server response.

        Args:
            english_msg: English version of client message
            reverse_map: Reverse mapping from client

        Returns:
            Tuple of (english_response, synthetic_response)
        """
        # Extract name, company, topic from message
        # Simple parsing: "Checking in, this is NAME with COMPANY talking about TOPIC"
        import re

        # Try to extract components
        pattern = r"(?:this is|I am) (\w+) (?:with|from) (\w+) (?:talking about|regarding) ([\w_]+)"
        match = re.search(pattern, english_msg)

        if match:
            name = match.group(1)
            company = match.group(2)
            topic = match.group(3)

            # Find concept IDs
            name_matches = self.concept_dict.search_by_term(name)
            company_matches = self.concept_dict.search_by_term(company)
            topic_matches = self.concept_dict.search_by_term(topic)

            if name_matches and company_matches and topic_matches:
                variables = {
                    'name': name_matches[0].id,
                    'company': company_matches[0].id,
                    'topic': topic_matches[0].id
                }

                english_response = f"Information received, confirmed you are {name} with {company} talking about {topic}"
                synthetic_response = self.translator.translate_to_synthetic(
                    english_response,
                    template_name='authentication_server',
                    variables=variables
                )

                return english_response, synthetic_response

        # Fallback: simple confirmation
        english_response = "Information received, authentication confirmed"
        synthetic_response = self.translator.translate_to_synthetic(english_response)

        return english_response, synthetic_response

    def run(self):
        """Run the server."""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold blue]GREMLIN Server[/bold blue]\n"
            f"[dim]Listening on: {self.host}:{self.port}[/dim]\n"
            f"[dim]Language: {self.pack.language_id[:16]}...[/dim]",
            box=box.DOUBLE
        ))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)

            self.console.print("[green]✅ Server ready, waiting for connections...[/green]\n")

            try:
                while True:
                    conn, addr = server_socket.accept()
                    # Handle each connection in current thread for demo simplicity
                    self.handle_client(conn, addr)

                    # Show stats
                    stats = self.pack.get_stats()
                    self.console.print(f"[dim]Language pack usage: {stats['usage_percentage']:.3f}%[/dim]\n")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Shutting down server...[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="GREMLIN Server Demo")
    parser.add_argument('--pack', '-p', type=Path, required=True, help="Language pack file")
    parser.add_argument('--host', default='0.0.0.0', help="Host to bind to")
    parser.add_argument('--port', type=int, default=9999, help="Port to listen on")
    parser.add_argument('--gemma', type=str, help="Path to Gemma 2 model")

    args = parser.parse_args()

    server = GremlinServer(args.pack, args.host, args.port, args.gemma)
    server.run()


if __name__ == '__main__':
    from typing import Optional
    main()
