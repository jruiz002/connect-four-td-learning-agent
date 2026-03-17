import numpy as np
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.heuristic import evaluate

_nodes_visited = 0


def minimax(game: Connect4, depth: int, maximizing: bool) -> tuple[int, int]:
    """
    Pure Minimax.
    Returns (best_score, nodes_visited_in_this_subtree).
    """
    global _nodes_visited
    _nodes_visited += 1

    if game.is_terminal() or depth == 0:
        if game.is_terminal():
            w = game.winner()
            if w == PLAYER:
                return 1_000_000, 1
            elif w == OPPONENT:
                return -1_000_000, 1
            else:  # Draw
                return 0, 1
        else:
            return evaluate(game.board), 1

    valid_cols = game.actions()
    nodes = 1  # count current node

    if maximizing:
        best = float("-inf")
        for col in valid_cols:
            child = game.result(col, PLAYER)
            score, child_nodes = minimax(child, depth - 1, False)
            nodes += child_nodes
            best = max(best, score)
        return best, nodes
    else:
        best = float("inf")
        for col in valid_cols:
            child = game.result(col, OPPONENT)
            score, child_nodes = minimax(child, depth - 1, True)
            nodes += child_nodes
            best = min(best, score)
        return best, nodes


def get_best_move(game: Connect4, depth: int = 4) -> tuple[int, int]:
    """
    Return (best_column, total_nodes_visited) using pure Minimax.
    AI plays as PLAYER (maximizer).
    """
    best_score = float("-inf")
    best_col = game.actions()[0]
    total_nodes = 0

    # Prefer center columns first (move ordering hint)
    cols = sorted(game.actions(), key=lambda c: abs(c - 3))

    for col in cols:
        child = game.result(col, PLAYER)
        score, nodes = minimax(child, depth - 1, False)
        total_nodes += nodes
        if score > best_score:
            best_score = score
            best_col = col

    return best_col, total_nodes
