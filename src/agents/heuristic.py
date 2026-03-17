import numpy as np
from src.core.connect4 import ROWS, COLS, PLAYER, OPPONENT, EMPTY


def _score_window(window: list[int], piece: int) -> int:
    """Score a 4-cell window for *piece*."""
    opp = OPPONENT if piece == PLAYER else PLAYER
    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp)

    score = 0
    if piece_count == 4:
        score += 1000
    elif piece_count == 3 and empty_count == 1:
        score += 50
    elif piece_count == 2 and empty_count == 2:
        score += 10

    if opp_count == 3 and empty_count == 1:
        score -= 80

    return score


def evaluate(board: np.ndarray) -> int:
    """
    Evaluate the board from PLAYER's (AI) perspective.
    Returns a positive score if the position is good for PLAYER,
    negative if good for OPPONENT.
    """
    score = 0

    # ---- Center column preference ----
    center_col = list(board[:, COLS // 2])
    center_count = center_col.count(PLAYER)
    score += center_count * 3

    # ---- Horizontal windows ----
    for r in range(ROWS):
        row = list(board[r, :])
        for c in range(COLS - 3):
            window = row[c:c + 4]
            score += _score_window(window, PLAYER)

    # ---- Vertical windows ----
    for c in range(COLS):
        col = list(board[:, c])
        for r in range(ROWS - 3):
            window = col[r:r + 4]
            score += _score_window(window, PLAYER)

    # ---- Diagonal ↘ windows ----
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += _score_window(window, PLAYER)

    # ---- Diagonal ↙ windows ----
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = [board[r + i][c - i] for i in range(4)]
            score += _score_window(window, PLAYER)

    return score
