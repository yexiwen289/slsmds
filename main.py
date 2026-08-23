import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game import _main_tui

if __name__ == "__main__":
    _main_tui()
