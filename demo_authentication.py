#!/usr/bin/env python3
"""
GREMLIN Authentication Demo
Simulates secure authentication between client and server using synthetic language.
"""

import random
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from core import ConceptDictionary, WordGenerator, LanguagePack, GrammarEngine


class AuthenticationDemo:
    """Demonstrates GREMLIN authentication protocol."""

    def __init__(self, language_pack: LanguagePack):
        self.pack = language_pack
        self.cd = ConceptDictionary()
        self.grammar = GrammarEngine()
        self.console = Console()

    def generate_authentication_exchange(self):
        """Generate one authentication round."""
        # Pick random authentication data
        names = self.cd.get_concepts_by_category('variables_names')
        companies = self.cd.get_concepts_by_category('variables_companies')
        topics = self.cd.get_concepts_by_category('variables_topics')

        name = random.choice(names)
        company = random.choice(companies)
        topic = random.choice(topics)

        # Generate exchange
        eng_client, syn_client, eng_server, syn_server = self.grammar.generate_authentication_pair(
            name.id, company.id, topic.id, self.pack, self.cd
        )

        return {
            'name': name.term,
            'company': company.term,
            'topic': topic.term,
            'english_client': eng_client,
            'synthetic_client': syn_client,
            'english_server': eng_server,
            'synthetic_server': syn_server
        }

    def run_demo(self, rounds: int = 5, show_mitm: bool = True):
        """
        Run the authentication demo.

        Args:
            rounds: Number of authentication rounds
            show_mitm: Show MITM perspective
        """
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold cyan]GREMLIN Authentication Protocol Demo[/bold cyan]\n"
            "[dim]Secure AI-to-AI authentication using ephemeral synthetic languages[/dim]",
            box=box.DOUBLE
        ))

        # Show language pack info
        stats = self.pack.get_stats()
        info_table = Table(show_header=False, box=box.SIMPLE)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Language ID", stats['language_id'][:16] + "...")
        info_table.add_row("Total Words", f"{stats['total_words']:,}")
        info_table.add_row("Grammar", self.pack.grammar.word_order)
        info_table.add_row("Rounds", str(rounds))

        self.console.print(Panel(info_table, title="Language Pack Info", border_style="cyan"))

        # Run authentication rounds
        for i in range(1, rounds + 1):
            exchange = self.generate_authentication_exchange()

            self.console.print(f"\n[bold yellow]━━━ Round {i}/{rounds} ━━━[/bold yellow]\n")

            # Client message
            client_table = Table(show_header=True, box=box.ROUNDED, border_style="green")
            client_table.add_column("CLIENT → SERVER", style="white")

            client_table.add_row(
                f"[dim]English:[/dim]\n{exchange['english_client']}\n\n"
                f"[bold green]Synthetic:[/bold green]\n{exchange['synthetic_client']}"
            )

            self.console.print(client_table)

            # MITM view (if enabled)
            if show_mitm:
                self.console.print(
                    f"[dim italic]  👁  MITM sees: {exchange['synthetic_client']}  (unintelligible)[/dim italic]"
                )

            # Server response
            server_table = Table(show_header=True, box=box.ROUNDED, border_style="blue")
            server_table.add_column("SERVER → CLIENT", style="white")

            server_table.add_row(
                f"[dim]English:[/dim]\n{exchange['english_server']}\n\n"
                f"[bold blue]Synthetic:[/bold blue]\n{exchange['synthetic_server']}"
            )

            self.console.print(server_table)

            # MITM view (if enabled)
            if show_mitm:
                self.console.print(
                    f"[dim italic]  👁  MITM sees: {exchange['synthetic_server']}  (unintelligible)[/dim italic]"
                )

        # Final statistics
        final_stats = self.pack.get_stats()

        stats_table = Table(show_header=False, box=box.SIMPLE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Words Used", f"{final_stats['total_used']:,}")
        stats_table.add_row("Words Remaining", f"{final_stats['words_remaining']:,}")
        stats_table.add_row("Usage", f"{final_stats['usage_percentage']:.3f}%")

        words_per_round = final_stats['total_used'] / rounds if rounds > 0 else 0
        remaining_rounds = int(final_stats['words_remaining'] / words_per_round) if words_per_round > 0 else 0
        stats_table.add_row("Est. Remaining Rounds", f"~{remaining_rounds:,}")

        self.console.print("\n")
        self.console.print(Panel(stats_table, title="Language Pack Usage", border_style="yellow"))

        # Security note
        self.console.print("\n")
        self.console.print(Panel(
            "[bold green]✓[/bold green] Authentication successful!\n\n"
            "[dim]Without the language pack, the synthetic messages are completely unintelligible.\n"
            "Each word is used only once (one-time pad), preventing pattern analysis.\n"
            "When words are depleted, simply generate and download a new language pack.[/dim]",
            title="Security Status",
            border_style="green"
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Demo GREMLIN authentication protocol"
    )

    parser.add_argument(
        '--rounds', '-r',
        type=int,
        default=5,
        help='Number of authentication rounds (default: 5)'
    )

    parser.add_argument(
        '--pack', '-p',
        type=Path,
        default=None,
        help='Language pack to use (default: generate new pack)'
    )

    parser.add_argument(
        '--no-mitm',
        action='store_true',
        help='Hide MITM perspective'
    )

    parser.add_argument(
        '--words',
        type=int,
        default=500,
        help='Words per concept for new pack (default: 500)'
    )

    args = parser.parse_args()

    console = Console()

    # Load or generate language pack
    if args.pack and args.pack.exists():
        console.print(f"[cyan]Loading language pack from {args.pack}...[/cyan]")
        pack = LanguagePack.load(args.pack)
    else:
        console.print(f"[cyan]Generating new language pack ({args.words} words/concept)...[/cyan]")
        cd = ConceptDictionary()
        wg = WordGenerator(
            min_length=4,
            max_length=12,
            use_blocks=['latin_basic', 'cyrillic', 'greek', 'arabic', 'hiragana', 'katakana']
        )
        pack = LanguagePack.generate(cd, words_per_concept=args.words, word_generator=wg)

    # Run demo
    demo = AuthenticationDemo(pack)
    demo.run_demo(rounds=args.rounds, show_mitm=not args.no_mitm)


if __name__ == '__main__':
    main()
