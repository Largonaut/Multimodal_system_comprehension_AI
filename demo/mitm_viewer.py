#!/usr/bin/env python3
"""
GREMLIN MITM Viewer (Laptop 3)
Displays intercepted synthetic language traffic (appears as gibberish).
"""

import argparse
import socket
import threading
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from rich.text import Text


class MITMViewer:
    """
    Man-in-the-Middle viewer for GREMLIN traffic.

    Shows intercepted messages but cannot understand them without the language pack.
    """

    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 9998,
        server_host: str = "localhost",
        server_port: int = 9999
    ):
        """
        Initialize MITM viewer.

        Args:
            listen_host: Host to listen on (for client connections)
            listen_port: Port to listen on
            server_host: Real server host
            server_port: Real server port
        """
        self.console = Console()
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.server_host = server_host
        self.server_port = server_port

        self.intercepted = []
        self.packet_count = 0

    def intercept(self, conn, addr):
        """
        Intercept a connection.

        Args:
            conn: Client connection
            addr: Client address
        """
        try:
            # Receive from client
            client_data = conn.recv(4096)

            if not client_data:
                return

            self.packet_count += 1

            # Display intercepted message
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Try to decode (will be gibberish)
            try:
                import json
                payload = json.loads(client_data.decode())
                message = payload.get('message', str(client_data))
            except:
                message = client_data.decode(errors='ignore')

            self.intercepted.append({
                'timestamp': timestamp,
                'direction': 'CLIENT → SERVER',
                'from': addr[0],
                'data': message,
                'size': len(client_data)
            })

            self._display_intercept()

            # Forward to real server (if in proxy mode)
            if self.server_host and self.server_port:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
                    server_sock.connect((self.server_host, self.server_port))
                    server_sock.sendall(client_data)

                    # Receive response
                    server_data = server_sock.recv(4096)

                    if server_data:
                        # Display intercepted response
                        try:
                            response_payload = json.loads(server_data.decode())
                            response_message = response_payload.get('message', str(server_data))
                        except:
                            response_message = server_data.decode(errors='ignore')

                        self.intercepted.append({
                            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            'direction': 'SERVER → CLIENT',
                            'from': self.server_host,
                            'data': response_message,
                            'size': len(server_data)
                        })

                        self._display_intercept()

                        # Forward response to client
                        conn.sendall(server_data)

        except Exception as e:
            self.console.print(f"[red]Error intercepting: {e}[/red]")
        finally:
            conn.close()

    def _display_intercept(self):
        """Display the latest intercepted packet."""
        packet = self.intercepted[-1]

        self.console.print(f"\n[yellow]━━━ Packet #{self.packet_count} Intercepted ━━━[/yellow]")

        table = Table(show_header=False, box=box.ROUNDED, border_style="red")
        table.add_column("Field", style="red")
        table.add_column("Value", style="white")

        table.add_row("Timestamp", packet['timestamp'])
        table.add_row("Direction", packet['direction'])
        table.add_row("From", packet['from'])
        table.add_row("Size", f"{packet['size']} bytes")
        table.add_row("Data", Text(packet['data'], style="bold yellow"))

        self.console.print(table)

        # Dramatic warning
        self.console.print(
            "[bold red]⚠️  WARNING: Message appears to be in unknown language/cipher[/bold red]"
        )
        self.console.print(
            "[dim red]   Cannot decrypt without language pack...[/dim red]\n"
        )

    def run_proxy_mode(self):
        """Run as active MITM proxy."""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold red]GREMLIN MITM Viewer - Proxy Mode[/bold red]\n"
            f"[dim]Listening: {self.listen_host}:{self.listen_port}[/dim]\n"
            f"[dim]Forwarding to: {self.server_host}:{self.server_port}[/dim]\n"
            f"[dim yellow]⚠️  Intercepting traffic...[/dim yellow]",
            box=box.DOUBLE
        ))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.listen_host, self.listen_port))
            server_socket.listen(5)

            self.console.print("[red]🔴 MITM Proxy active, waiting for traffic...[/red]\n")

            try:
                while True:
                    conn, addr = server_socket.accept()
                    self.intercept(conn, addr)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Stopping MITM viewer...[/yellow]")

    def run_passive_mode(self, pcap_file: Path = None):
        """
        Run in passive mode (display pre-captured or simulated traffic).

        Args:
            pcap_file: Optional pcap file to read (not implemented yet)
        """
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold red]GREMLIN MITM Viewer - Passive Mode[/bold red]\n"
            "[dim]Displaying captured traffic...[/dim]",
            box=box.DOUBLE
        ))

        # Simulate some intercepted traffic for demo
        simulated_traffic = [
            "ポΉlڡ؜o]ӥTϑ エsπζϗ ギ؃ッ΂ϖCOボぜз",
            "ЭヾЧ؇リeЇBサ΄ېだ ZӢがٮらAи әゕEϖずڛ٬フح",
            "ゴكφへメۘ`ぶҴ ӥaるのゎۍ ちエӌϿユШゑ",
            "ұrϰѹϒПヨМ ザڴؑΧぽέペ] νіつベヸぜー",
        ]

        for i, msg in enumerate(simulated_traffic, 1):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            direction = "CLIENT → SERVER" if i % 2 == 1 else "SERVER → CLIENT"

            packet = {
                'timestamp': timestamp,
                'direction': direction,
                'from': '192.168.1.100' if i % 2 == 1 else '192.168.1.200',
                'data': msg,
                'size': len(msg.encode())
            }

            self.intercepted.append(packet)
            self.packet_count = i

            self._display_intercept()

            import time
            time.sleep(2)

        self.console.print("\n[green]End of captured traffic[/green]")


def main():
    parser = argparse.ArgumentParser(description="GREMLIN MITM Viewer")
    parser.add_argument('--mode', choices=['proxy', 'passive'], default='passive',
                        help="Viewing mode")
    parser.add_argument('--listen-host', default='0.0.0.0', help="Host to listen on")
    parser.add_argument('--listen-port', type=int, default=9998, help="Port to listen on")
    parser.add_argument('--server-host', default='localhost', help="Server to forward to")
    parser.add_argument('--server-port', type=int, default=9999, help="Server port")

    args = parser.parse_args()

    viewer = MITMViewer(
        args.listen_host,
        args.listen_port,
        args.server_host,
        args.server_port
    )

    if args.mode == 'proxy':
        viewer.run_proxy_mode()
    else:
        viewer.run_passive_mode()


if __name__ == '__main__':
    main()
