"""
game.py
-------
Terminal game runner for Connect Four with colorized board display.

Requires: colorama  →  pip install colorama
"""

import time
from src.core.connect4 import Connect4, PLAYER, OPPONENT, ROWS, COLS

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _FakeStyle:
        def __getattr__(self, name):
            return ""

    Fore = _FakeStyle()
    Style = _FakeStyle()


# ── Display helpers ──────────────────────────────────────────────────────────

PIECE_COLOR = {
    PLAYER:   Fore.RED    if HAS_COLOR else "",
    OPPONENT: Fore.YELLOW if HAS_COLOR else "",
}
PIECE_SYMBOL = {0: "·", PLAYER: "●", OPPONENT: "●"}


def _piece_str(val: int) -> str:
    color = PIECE_COLOR.get(val, "")
    symbol = PIECE_SYMBOL.get(val, "·")
    reset = Style.RESET_ALL if HAS_COLOR else ""
    return f"{color}{symbol}{reset}"


def print_board(game: Connect4) -> None:
    border = "─" * (COLS * 2 + 1)
    col_header = "  " + "  ".join(str(c) for c in range(COLS))
    print(f"\n{col_header}")
    print(f"┌{border}┐")
    for r in range(ROWS):
        row_str = " │ " + "  ".join(_piece_str(game.board[r][c]) for c in range(COLS)) + " │"
        print(row_str)
    print(f"└{border}┘")


def print_separator(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


# ── Game runner ──────────────────────────────────────────────────────────────

class GameRunner:
    """
    Runs a full Connect Four game between two agents.

    agent1 → plays as PLAYER (1) – X / Red
    agent2 → plays as OPPONENT (2) – O / Yellow
    """

    def __init__(self, agent1, agent2, verbose: bool = True, delay: float = 0.4):
        self.agent1 = agent1
        self.agent2 = agent2
        self.verbose = verbose
        self.delay = delay  # seconds between AI moves (for readability)

    def run(self) -> int | None:
        """
        Play a game to completion.
        Returns: PLAYER (1) if agent1 wins, OPPONENT (2) if agent2 wins, 0 for draw.
        """
        game = Connect4()
        turn = PLAYER  # PLAYER goes first

        if self.verbose:
            print_separator("NEW GAME")
            print(f"  {Fore.RED if HAS_COLOR else ''}■ {self.agent1.name}{Style.RESET_ALL if HAS_COLOR else ''} (X / Red) "
                  f"  vs  "
                  f"{Fore.YELLOW if HAS_COLOR else ''}■ {self.agent2.name}{Style.RESET_ALL if HAS_COLOR else ''} (O / Yellow)")
            print_board(game)

        while not game.is_terminal():
            if turn == PLAYER:
                agent = self.agent1
                piece = PLAYER
                color = Fore.RED if HAS_COLOR else ""
            else:
                agent = self.agent2
                piece = OPPONENT
                color = Fore.YELLOW if HAS_COLOR else ""

            symbol = "X" if piece == PLAYER else "O"
            reset = Style.RESET_ALL if HAS_COLOR else ""

            if self.verbose:
                print(f"\n  {color}[{symbol}] {agent.name}'s turn…{reset}")

            t0 = time.perf_counter()
            col = agent.get_move(game)
            elapsed = time.perf_counter() - t0

            game = game.result(col, piece)

            if self.verbose:
                nodes_info = ""
                if hasattr(agent, "_last_nodes") and agent._last_nodes:
                    nodes_info = f"  |  nodes: {agent._last_nodes:,}"
                print(f"  → Column {col}  ({elapsed:.3f}s{nodes_info})")
                print_board(game)
                if hasattr(agent, "name") and "Alpha" not in agent.name:
                    time.sleep(self.delay)

            turn = OPPONENT if turn == PLAYER else PLAYER

        # ── Game over ────────────────────────────────────────────────────────
        winner = game.winner()
        if self.verbose:
            print_separator("GAME OVER")
            if winner == PLAYER:
                color = Fore.RED if HAS_COLOR else ""
                reset = Style.RESET_ALL if HAS_COLOR else ""
                print(f"  🏆  {color}{self.agent1.name}{reset} WINS!\n")
            elif winner == OPPONENT:
                color = Fore.YELLOW if HAS_COLOR else ""
                reset = Style.RESET_ALL if HAS_COLOR else ""
                print(f"  🏆  {color}{self.agent2.name}{reset} WINS!\n")
            else:
                print("  🤝  It's a DRAW!\n")

        return winner if winner else 0
