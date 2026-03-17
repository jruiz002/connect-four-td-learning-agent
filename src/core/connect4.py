"""
connect4.py
-----------
Core Connect Four game logic.
Board values: 0 = empty, 1 = Player / Maximizer (AI), 2 = Opponent / Minimizer.
"""

import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1   # AI / Maximizer
OPPONENT = 2  # Human / Random / Minimizer


class Connect4:
    """Manages the state of a Connect Four board."""

    def __init__(self, board: np.ndarray | None = None):
        if board is None:
            self.board = np.zeros((ROWS, COLS), dtype=int)
        else:
            self.board = board.copy()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def copy(self) -> "Connect4":
        return Connect4(self.board)

    def actions(self) -> list[int]:
        """Return list of valid column indices (columns that are not full)."""
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def _drop_row(self, col: int) -> int:
        """Return the lowest empty row index for a given column."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                return row
        return -1  # column full

    def result(self, col: int, piece: int) -> "Connect4":
        """Return a new Connect4 state after dropping *piece* into *col*."""
        new_game = self.copy()
        row = new_game._drop_row(col)
        if row == -1:
            raise ValueError(f"Column {col} is full.")
        new_game.board[row][col] = piece
        return new_game

    # ------------------------------------------------------------------
    # Win / terminal detection
    # ------------------------------------------------------------------

    def _check_window(self, window: np.ndarray, piece: int) -> bool:
        """Return True if *window* contains 4 consecutive *piece* values."""
        return list(window).count(piece) == 4

    def winner(self) -> int | None:
        """Return PLAYER (1), OPPONENT (2), or None if no winner yet."""
        board = self.board

        for piece in (PLAYER, OPPONENT):
            # Horizontal
            for r in range(ROWS):
                for c in range(COLS - 3):
                    if self._check_window(board[r, c:c + 4], piece):
                        return piece

            # Vertical
            for r in range(ROWS - 3):
                for c in range(COLS):
                    if self._check_window(board[r:r + 4, c], piece):
                        return piece

            # Diagonal ↘
            for r in range(ROWS - 3):
                for c in range(COLS - 3):
                    window = [board[r + i][c + i] for i in range(4)]
                    if self._check_window(np.array(window), piece):
                        return piece

            # Diagonal ↙
            for r in range(ROWS - 3):
                for c in range(3, COLS):
                    window = [board[r + i][c - i] for i in range(4)]
                    if self._check_window(np.array(window), piece):
                        return piece

        return None

    def is_draw(self) -> bool:
        """Return True if the board is full and there is no winner."""
        return len(self.actions()) == 0 and self.winner() is None

    def is_terminal(self) -> bool:
        """Return True if the game is over (win or draw)."""
        return self.winner() is not None or self.is_draw()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        symbols = {EMPTY: ".", PLAYER: "X", OPPONENT: "O"}
        rows = []
        for r in range(ROWS):
            rows.append(" ".join(symbols[self.board[r][c]] for c in range(COLS)))
        header = " ".join(str(c) for c in range(COLS))
        return header + "\n" + "\n".join(rows)
