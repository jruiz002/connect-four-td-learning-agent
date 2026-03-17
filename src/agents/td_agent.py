"""
td_agent.py
-----------
Task 2.1 - TD Learning Agent for Connect Four using Q-Learning.
"""

import random
import pickle
import numpy as np
from pathlib import Path
from src.core.connect4 import Connect4, PLAYER, OPPONENT

class TDAgent:
    """
    Q-Learning (off-policy) agent for Connect Four.
    Maintains a tabular Q-function mapping (state, action) -> value.
    """

    def __init__(self, piece: int = PLAYER, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.piece = piece
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = f"TD Agent (Q-Learning)"
        
        # State: Tuple of tuples representing grid -> Dict of {action: Q-value}
        self.q_table: dict[tuple, dict[int, float]] = {}

    def _get_state_key(self, game: Connect4) -> tuple:
        """Convert the numpy board to a hashable tuple of tuples."""
        # To generalize, we could normalize the pieces so the agent always
        # sees itself as '1' and opponent as '2'. Since we might play as PLAYER
        # or OPPONENT, we normalize the board representation.
        board = game.board
        if self.piece == OPPONENT:
            # Swap 1s and 2s if we are playing as OPPONENT
            board = board.copy()
            board[game.board == PLAYER] = OPPONENT
            board[game.board == OPPONENT] = PLAYER

        return tuple(tuple(row) for row in board)

    def _get_q_value(self, state_key: tuple, action: int) -> float:
        """Return Q(s, a). Initialize to 0.0 if unseen."""
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key].get(action, 0.0)

    def _get_max_q(self, game: Connect4) -> float:
        """Return max_a Q(s, a) for a given state."""
        actions = game.actions()
        if not actions:
            return 0.0
        state_key = self._get_state_key(game)
        return max(self._get_q_value(state_key, a) for a in actions)

    def get_move(self, game: Connect4, training: bool = False) -> int:
        """
        Choose an action using epsilon-greedy policy.
        When not training, it acts greedily (epsilon = 0).
        """
        actions = game.actions()
        if not actions:
            raise ValueError("No valid moves available.")

        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation
        state_key = self._get_state_key(game)
        q_values = [self._get_q_value(state_key, a) for a in actions]
        
        max_q = max(q_values)
        # In case of ties, choose randomly among the best moves
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state: Connect4, action: int, reward: float, next_state: Connect4):
        """
        Q-Learning update rule:
        Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        state_key = self._get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            
        current_q = self._get_q_value(state_key, action)
        
        # If terminal state, max Q(s', a') is 0
        if next_state.is_terminal():
            max_next_q = 0.0
        else:
            max_next_q = self._get_max_q(next_state)

        # Off-policy update (Q-Learning uses max over next actions regardless of policy)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    # --- Persistence ---
    def save(self, filepath: str = "td_agent.pkl"):
        """Save the Q-table to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath: str = "td_agent.pkl"):
        """Load the Q-table from a pickle file."""
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
        else:
            print(f"No existing Q-table found at {filepath}. Starting fresh.")
