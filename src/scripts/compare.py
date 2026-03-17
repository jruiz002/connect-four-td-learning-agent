"""
compare.py
----------
Task 2.2 – Demonstration: Minimax vs Alpha-Beta node count comparison.

Loads a mid-game board state and runs both algorithms at depth 4,
printing the number of nodes each visited.
"""

import numpy as np
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.minimax import get_best_move
from src.agents.alphabeta import get_best_move_ab

# ── Mid-game board state for fair comparison ────────────────────────────────
# Rows 0-5 (top to bottom), columns 0-6 (left to right)
# 0 = empty, 1 = AI (X), 2 = Opponent (O)
MID_GAME_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 2, 2, 1, 0, 0, 0],
    [2, 1, 1, 2, 1, 0, 0],
], dtype=int)


def run_comparison(depth: int = 4) -> None:
    print("=" * 60)
    print("  MINIMAX vs ALPHA-BETA PRUNING  –  Node Count Comparison")
    print("=" * 60)

    game = Connect4(MID_GAME_BOARD)
    print("\nBoard state used for comparison:")
    print(game)
    print(f"\nDepth = {depth}\n")

    # ── Pure Minimax ─────────────────────────────────────────────────────────
    import time
    t0 = time.perf_counter()
    mm_col, mm_nodes = get_best_move(game, depth)
    mm_time = time.perf_counter() - t0

    print(f"  Minimax Pure:")
    print(f"    Best column : {mm_col}")
    print(f"    Nodes visited: {mm_nodes:,}")
    print(f"    Time elapsed : {mm_time:.4f}s")

    # ── Alpha-Beta ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    ab_col, ab_nodes = get_best_move_ab(game, depth)
    ab_time = time.perf_counter() - t0

    print(f"\n  Alpha-Beta Pruning:")
    print(f"    Best column : {ab_col}")
    print(f"    Nodes visited: {ab_nodes:,}")
    print(f"    Time elapsed : {ab_time:.4f}s")

    # ── Comparison ───────────────────────────────────────────────────────────
    reduction = (1 - ab_nodes / mm_nodes) * 100 if mm_nodes > 0 else 0
    speedup   = mm_time / ab_time if ab_time > 0 else float("inf")

    print("\n" + "-" * 60)
    print(f"  Node reduction : {reduction:.1f}%  ({mm_nodes:,} → {ab_nodes:,})")
    print(f"  Speedup factor : {speedup:.2f}×")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_comparison(depth=4)
