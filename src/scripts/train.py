"""
train.py
--------
Task 2.1 - Training loop for the TD Learning Agent.
"""

import time
import random
from src.core.connect4 import Connect4, PLAYER, OPPONENT
from src.agents.td_agent import TDAgent
from src.agents.agents import RandomAgent

def train(episodes: int = 50000, save_path: str = "td_agent.pkl"):
    """
    Train the TD Agent by playing against a Random Agent.
    """
    print(f"Starting training for {episodes} episodes...")
    
    agent = TDAgent(piece=PLAYER, alpha=0.1, gamma=0.9, epsilon=1.0)
    agent.load(save_path) # Load if exists to continue training
    
    opponent = RandomAgent(piece=OPPONENT)
    
    # Epsilon decay parameters
    epsilon_start = agent.epsilon
    epsilon_min = 0.1
    epsilon_decay = (epsilon_start - epsilon_min) / (episodes * 0.8) # Decay over 80% of training

    wins = 0
    losses = 0
    draws = 0
    
    t0 = time.time()
    
    for episode in range(1, episodes + 1):
        game = Connect4()
        
        # Decide who goes first randomly to generalize learning
        turn = PLAYER if random.random() < 0.5 else OPPONENT
        
        # Keep track of the last state and action for the TD Agent to update
        last_state = None
        last_action = None
        
        while not game.is_terminal():
            if turn == PLAYER:
                state_before_move = game.copy()
                action = agent.get_move(game, training=True)
                
                game = game.result(action, PLAYER)
                
                # If the agent just made a move that ended the game, we update immediately.
                if game.is_terminal():
                    winner = game.winner()
                    if winner == PLAYER:
                        reward = 1.0
                        wins += 1
                    elif winner == OPPONENT:
                        # Should not happen if it's PLAYER's turn, but just in case
                        reward = -1.0
                        losses += 1
                    else:
                        reward = 0.5 # Draw
                        draws += 1
                    
                    agent.update(state_before_move, action, reward, game)
                else:
                    # If game continues, we wait for opponent's move to see the outcome
                    # before updating this step (though technically in standard Q-learning
                    # against an environment we'd update immediately with reward 0 and the
                    # state *after* opponent's turn as the next state).
                    # For a 2-player game, next_state for PLAYER is the state AFTER OPPONENT moves.
                    last_state = state_before_move
                    last_action = action
                    
            else:
                # Opponent's turn
                action_opp = opponent.get_move(game)
                game = game.result(action_opp, OPPONENT)
                
                # If this is not the first turn of the game, PLAYER has a pending update
                if last_state is not None:
                    if game.is_terminal():
                        winner = game.winner()
                        if winner == OPPONENT:
                            reward = -1.0
                            losses += 1
                        elif winner == PLAYER:
                            reward = 1.0 # Very unlikely that opp causes player to win directly
                            wins += 1
                        else:
                            reward = 0.5 # Draw
                            draws += 1
                    else:
                        # Intermediate state: reward is 0
                        reward = 0.0
                    
                    agent.update(last_state, last_action, reward, game)
                    last_state = None
                    last_action = None

            turn = OPPONENT if turn == PLAYER else PLAYER
            
        # Decay epsilon
        if agent.epsilon > epsilon_min:
            agent.epsilon -= epsilon_decay

        # Progress reporting
        if episode % 5000 == 0:
            elapsed = time.time() - t0
            win_rate = wins / 5000 * 100
            print(f"Episode {episode}/{episodes} | Win Rate: {win_rate:.1f}% | Epsilon: {agent.epsilon:.3f} | States: {len(agent.q_table)} | Time: {elapsed:.1f}s")
            wins, losses, draws = 0, 0, 0
            t0 = time.time()
            agent.save(save_path) # Save checkpoint

    # Final save
    agent.save(save_path)
    print("Training finished!")
    print(f"Total states learned: {len(agent.q_table)}")

if __name__ == "__main__":
    import random
    train(episodes=50000)
    
