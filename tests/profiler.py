import pyinstrument

from Chess.board import Board
from Chess.chessEngine import GameState

board = Board.new_from_fen("r1q4r/1n2kn2/1b4b1/1p1p1p1p/1P1P1P1P/1B4B1/1N2KN2/R1Q4R w - - 0 1")
game_state = GameState()
game_state.board = [
    ["bR", "--", "bQ", "--", "--", "--", "--", "bR"],
    ["--", "bN", "--", "--", "bK", "bN", "--", "--"],
    ["--", "bB", "--", "--", "--", "--", "bB", "--"],
    ["--", "bP", "--", "bP", "--", "bP", "--", "bP"],
    ["--", "wP", "--", "wP", "--", "wP", "--", "wP"],
    ["--", "wB", "--", "--", "--", "--", "wB", "--"],
    ["--", "wN", "--", "--", "wK", "wN", "--", "--"],
    ["wR", "--", "wQ", "--", "--", "--", "--", "wR"],
]

print('running profiler')

# Create pyinstrument Profiler
profiler = pyinstrument.Profiler()

# Profile get_all_valid_moves method
profiler.start()

for _ in range(1000):
    board.get_all_valid_moves()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
# profiler.open_in_browser()

# Profile get_valid_moves method
profiler.start()

for _ in range(1000):
    game_state.get_valid_moves()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
