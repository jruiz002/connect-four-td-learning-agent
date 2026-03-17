"""
agents.py
---------
Agent classes for Connect Four:
  - RandomAgent    : picks a random valid column
  - HumanAgent     : reads column choice from stdin
  - AlphaBetaAgent : uses Alpha-Beta pruning at a configurable depth
"""

import random
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.alphabeta import get_best_move_ab


class RandomAgent:
    """Picks a random valid column."""

    def __init__(self, piece: int = OPPONENT):
        self.piece = piece
        self.name = "Random Agent"

    def get_move(self, game: Connect4) -> int:
        valid = game.actions()
        return random.choice(valid)


class HumanAgent:
    """Reads a column from stdin."""

    def __init__(self, piece: int = OPPONENT):
        self.piece = piece
        self.name = "Human Player"

    def get_move(self, game: Connect4) -> int:
        valid = game.actions()
        while True:
            try:
                col = int(input(f"  Your move – choose a column {valid}: "))
                if col in valid:
                    return col
                print(f"  ✗ Column {col} is not available. Try again.")
            except ValueError:
                print("  ✗ Please enter a valid integer.")


class AlphaBetaAgent:
    """Uses Alpha-Beta pruning to pick the best move."""

    def __init__(self, piece: int = PLAYER, depth: int = 6):
        self.piece = piece
        self.depth = depth
        self.name = f"Alpha-Beta Agent (depth={depth})"
        self._last_nodes = 0

    def get_move(self, game: Connect4) -> int:
        # Temporarily swap perspective if AI plays as OPPONENT
        col, nodes = get_best_move_ab(game, self.depth)
        self._last_nodes = nodes
        return col
