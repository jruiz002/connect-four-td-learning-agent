"""
evaluate.py
-----------
Task 2.2 - Tournament comparison:
A) TD Agent vs Minimax
B) TD Agent vs AlphaBeta
C) Minimax vs AlphaBeta

Generates a visualization of the results (results.pdf).
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.agents import AlphaBetaAgent
from src.agents.td_agent import TDAgent

# Assuming the user has a pure Minimax agent (maybe just AlphaBeta with no pruning or
# we can use the `get_best_move` from minimax.py if available, wrapping it).
from src.agents.minimax import get_best_move

class MinimaxAgent:
    """Wrapper for the pure Minimax algorithm."""
    def __init__(self, piece: int = PLAYER, depth: int = 4):
        self.piece = piece
        self.depth = depth
        self.name = f"Minimax Agent (depth={depth})"

    def get_move(self, game: Connect4) -> int:
        col, _ = get_best_move(game, self.depth)
        return col


def play_match(agent1, agent2) -> int:
    """Plays a single game. Returns: 1 if agent1 wins, 2 if agent2 wins, 0 for draw."""
    game = Connect4()
    turn = PLAYER

    while not game.is_terminal():
        if turn == PLAYER:
            # Assumes the agents respect game.actions()
            col = agent1.get_move(game)
            game = game.result(col, PLAYER)
        else:
            col = agent2.get_move(game)
            game = game.result(col, OPPONENT)

        turn = OPPONENT if turn == PLAYER else PLAYER

    winner = game.winner()
    if winner == PLAYER:
        return 1
    elif winner == OPPONENT:
        return 2
    else:
        return 0


def run_condition(name: str, agent_a_class, agent_a_kwargs, agent_b_class, agent_b_kwargs, games: int = 50):
    print(f"\nRunning Condición {name}...")
    results = {"Win A": 0, "Win B": 0, "Draw": 0}

    for i in tqdm(range(games)):
        # Alternate who plays as PLAYER (who goes first)
        if i % 2 == 0:
            agent1 = agent_a_class(piece=PLAYER, **agent_a_kwargs)
            agent2 = agent_b_class(piece=OPPONENT, **agent_b_kwargs)
            res = play_match(agent1, agent2)
            if res == 1:
                results["Win A"] += 1
            elif res == 2:
                results["Win B"] += 1
            else:
                results["Draw"] += 1
        else:
            # agent2 plays first
            agent1 = agent_b_class(piece=PLAYER, **agent_b_kwargs)
            agent2 = agent_a_class(piece=OPPONENT, **agent_a_kwargs)
            res = play_match(agent1, agent2)
            # If agent1 (B) won, Win B increases
            if res == 1:
                results["Win B"] += 1
            elif res == 2:
                results["Win A"] += 1
            else:
                results["Draw"] += 1

    return results


def main():
    # Attempt to load the trained TD Agent first to ensure it's loaded
    print("Loading TD Agent model...")
    td_agent_template = TDAgent(piece=PLAYER, epsilon=0.0) # Greedy for evaluation
    td_agent_template.load("td_agent.pkl")
    # We will pass the pre-loaded table to new instances to avoid reloading
    q_table = td_agent_template.q_table
    if not q_table:
        print("Warning: td_agent.pkl empty or not found. Did you run train.py first?")

    class LoadedTDAgent(TDAgent):
        def __init__(self, piece: int = PLAYER, **kwargs):
            super().__init__(piece=piece, epsilon=0.0, **kwargs)
            self.q_table = q_table

    # Configuration 
    GAMES_PER_COND = 50
    DEPTH_MINIMAX = 3 # Kept low for execution time during 50 games
    DEPTH_ALPHABETA = 4

    print(f"Tournament config: {GAMES_PER_COND} games per condition.")
    print("Agent details:")
    print(f" - TD Agent (Q-Learning, states: {len(q_table)})")
    print(f" - Minimax (Depth {DEPTH_MINIMAX})")
    print(f" - AlphaBeta (Depth {DEPTH_ALPHABETA})")

    # Condición A: TD vs Minimax
    results_A = run_condition("A (TD vs Minimax)", 
                              LoadedTDAgent, {}, 
                              MinimaxAgent, {"depth": DEPTH_MINIMAX}, 
                              GAMES_PER_COND)

    # Condición B: TD vs AlphaBeta
    results_B = run_condition("B (TD vs AlphaBeta)", 
                              LoadedTDAgent, {}, 
                              AlphaBetaAgent, {"depth": DEPTH_ALPHABETA}, 
                              GAMES_PER_COND)

    # Condición C: Minimax vs AlphaBeta
    results_C = run_condition("C (Minimax vs AlphaBeta)", 
                              MinimaxAgent, {"depth": DEPTH_MINIMAX}, 
                              AlphaBetaAgent, {"depth": DEPTH_ALPHABETA}, 
                              GAMES_PER_COND)

    print("\n--- Tournament Results ---")
    print(f"Cond A (TD vs MM)  : TD Wins: {results_A['Win A']}, MM Wins: {results_A['Win B']}, Draws: {results_A['Draw']}")
    print(f"Cond B (TD vs AB)  : TD Wins: {results_B['Win A']}, AB Wins: {results_B['Win B']}, Draws: {results_B['Draw']}")
    print(f"Cond C (MM vs AB)  : MM Wins: {results_C['Win A']}, AB Wins: {results_C['Win B']}, Draws: {results_C['Draw']}")

    # Plotting
    labels = ['A: TD vs Minimax', 'B: TD vs Alpha-Beta', 'C: Minimax vs Alpha-Beta']
    
    agent_1_wins = [results_A['Win A'], results_B['Win A'], results_C['Win A']]
    agent_2_wins = [results_A['Win B'], results_B['Win B'], results_C['Win B']]
    draws = [results_A['Draw'], results_B['Draw'], results_C['Draw']]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, agent_1_wins, width, label='Agent 1 Wins (TD / TD / Minimax)', color='skyblue')
    rects2 = ax.bar(x, agent_2_wins, width, label='Agent 2 Wins (MM / AB / AB)', color='lightcoral')
    rects3 = ax.bar(x + width, draws, width, label='Draws', color='lightgreen')

    ax.set_ylabel('Number of Games')
    ax.set_title(f'Connect Four Agent Competition Results ({GAMES_PER_COND} games/condition)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    import os
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/results.pdf')
    print("\nPlot saved as 'output/results.pdf'.")

if __name__ == "__main__":
    main()
