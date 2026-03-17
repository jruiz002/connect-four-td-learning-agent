"""
alphabeta.py
------------
Alpha-Beta Pruning optimization of Minimax for Connect Four (Task 2.2).

PLAYER (1) is the maximizer, OPPONENT (2) is the minimizer.
Supports depth 5 or 6 efficiently.
"""

import numpy as np
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.heuristic import evaluate


def alphabeta(
    game: Connect4,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> tuple[int, int]:
    """
    Minimax with Alpha-Beta pruning.
    Returns (best_score, nodes_visited_in_this_subtree).
    """
    if game.is_terminal() or depth == 0:
        if game.is_terminal():
            w = game.winner()
            if w == PLAYER:
                return 1_000_000 + depth, 1   # prefer faster wins
            elif w == OPPONENT:
                return -1_000_000 - depth, 1  # prefer slower losses
            else:
                return 0, 1
        else:
            return evaluate(game.board), 1

    # Center-biased move ordering improves pruning efficiency
    valid_cols = sorted(game.actions(), key=lambda c: abs(c - 3))
    nodes = 1

    if maximizing:
        best = float("-inf")
        for col in valid_cols:
            child = game.result(col, PLAYER)
            score, child_nodes = alphabeta(child, depth - 1, alpha, beta, False)
            nodes += child_nodes
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # β cut-off
        return best, nodes
    else:
        best = float("inf")
        for col in valid_cols:
            child = game.result(col, OPPONENT)
            score, child_nodes = alphabeta(child, depth - 1, alpha, beta, True)
            nodes += child_nodes
            best = min(best, score)
            beta = min(beta, best)
            if alpha >= beta:
                break  # α cut-off
        return best, nodes


def get_best_move_ab(game: Connect4, depth: int = 6) -> tuple[int, int]:
    """
    Return (best_column, total_nodes_visited) using Alpha-Beta pruning.
    AI plays as PLAYER (maximizer).
    """
    best_score = float("-inf")
    best_col = game.actions()[0]
    total_nodes = 0
    alpha = float("-inf")
    beta = float("inf")

    cols = sorted(game.actions(), key=lambda c: abs(c - 3))

    for col in cols:
        child = game.result(col, PLAYER)
        score, nodes = alphabeta(child, depth - 1, alpha, beta, False)
        total_nodes += nodes
        if score > best_score:
            best_score = score
            best_col = col
        alpha = max(alpha, best_score)

    return best_col, total_nodes
