"""
main.py
-------
Entry point for Connect Four AI demos.

Menu:
  1. Node count demo        (Minimax vs Alpha-Beta comparison)
  2. AI vs Random           (AI should win consistently)
  3. AI vs Human            (interactive play)
  4. Run all automated demos
"""

import sys
from src.agents.agents import AlphaBetaAgent, RandomAgent, HumanAgent
from src.core.game import GameRunner, print_separator
from src.scripts.compare import run_comparison
from src.core.connect4 import PLAYER, OPPONENT
from src.agents.td_agent import TDAgent
from src.scripts.evaluate import MinimaxAgent

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


BANNER = r"""
  ____                            _     _  _   
 / ___| ___  _ __  _ __   ___  ___| |_  | || |  
| |    / _ \| '_ \| '_ \ / _ \/ __| __|  | || |_ 
| |___| (_) | | | | | | |  __/ (__| |_   |__   _|
 \____|\___/|_| |_|_| |_|\___|\___|\__|     |_|   

         Minimax + Alpha-Beta AI  –  Connect Four
"""

MENU = """
  ┌─────────────────────────────────────────┐
  │  1. Node Count Demo (Minimax vs AB)     │
  │  2. AI vs Random Agent                  │
  │  3. AI vs Human (you play!)             │
  │  4. Run All Automated Demos             │
  │  5. Video Recording Helper (Task 2.3)   │
  │  0. Exit                                │
  └─────────────────────────────────────────┘
"""


def demo_node_count() -> None:
    print_separator("TASK 2.2 – Node Count Comparison")
    run_comparison(depth=4)


def demo_ai_vs_random(n_games: int = 3) -> None:
    print_separator("TASK 2.3 – AI vs Random Agent")
    ai_wins = 0
    for i in range(1, n_games + 1):
        print(f"\n  ── Game {i}/{n_games} ──")
        ai_agent  = AlphaBetaAgent(piece=PLAYER,   depth=6)
        rnd_agent = RandomAgent(piece=OPPONENT)
        runner = GameRunner(ai_agent, rnd_agent, verbose=True, delay=0)
        result = runner.run()
        if result == PLAYER:
            ai_wins += 1

    print(f"\n  Result: AI won {ai_wins}/{n_games} games against Random Agent.")


def demo_ai_vs_human() -> None:
    print_separator("AI vs Human")
    ai_agent    = AlphaBetaAgent(piece=PLAYER,   depth=6)
    human_agent = HumanAgent(piece=OPPONENT)
    runner = GameRunner(ai_agent, human_agent, verbose=True, delay=0)
    runner.run()


def demo_video_recording() -> None:
    print_separator("Task 2.3: Video Recording Helper")
    print("This script will run exactly 1 game for each condition.")
    print("Take your time to explain the concepts during the matches.")
    
    td_agent_template = TDAgent(piece=PLAYER, epsilon=0.0)
    td_agent_template.load("td_agent.pkl")
    q_table = td_agent_template.q_table
    
    if not q_table:
        print("WARNING: TD Agent has not been trained (td_agent.pkl not found).")

    class LoadedTDAgent(TDAgent):
        def __init__(self, piece: int = PLAYER, **kwargs):
            super().__init__(piece=piece, epsilon=0.0, **kwargs)
            self.q_table = q_table

    input("\nPress ENTER when you are ready to start Condition A (TD vs Minimax)...")
    agent1 = LoadedTDAgent(piece=PLAYER)
    agent2 = MinimaxAgent(piece=OPPONENT, depth=3)
    print_separator("COND A: TD Learning (Red) vs Minimax Depth 3 (Yellow)")
    runner = GameRunner(agent1, agent2, verbose=True, delay=0.5)
    runner.run()

    input("\nGreat! Press ENTER when you are ready to start Condition B (TD vs AlphaBeta)...")
    agent1 = AlphaBetaAgent(piece=PLAYER, depth=4)
    agent2 = LoadedTDAgent(piece=OPPONENT)
    print_separator("COND B: AlphaBeta Depth 4 (Red) vs TD Learning (Yellow)")
    runner = GameRunner(agent1, agent2, verbose=True, delay=0.5)
    runner.run()
    
    input("\nPerfect. Press ENTER to start the control Condition C (Minimax vs AlphaBeta)...")
    agent1 = MinimaxAgent(piece=PLAYER, depth=3)
    agent2 = AlphaBetaAgent(piece=OPPONENT, depth=4)
    print_separator("COND C: Minimax Depth 3 (Red) vs AlphaBeta Depth 4 (Yellow)")
    runner = GameRunner(agent1, agent2, verbose=True, delay=0.5)
    runner.run()


def main() -> None:
    print(BANNER)

    while True:
        print(MENU)
        choice = input("  Select an option: ").strip()

        if choice == "1":
            demo_node_count()
        elif choice == "2":
            demo_ai_vs_random(n_games=3)
        elif choice == "3":
            demo_ai_vs_human()
        elif choice == "4":
            demo_node_count()
            demo_ai_vs_random(n_games=2)
        elif choice == "5":
            demo_video_recording()
        elif choice == "0":
            print("\n  Goodbye!\n")
            sys.exit(0)
        else:
            print("  ✗ Invalid option. Try again.")


if __name__ == "__main__":
    main()
