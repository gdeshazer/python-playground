import time
import unittest

from Chess.board import Board
from Chess.chessEngine import GameState


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.new_board = Board.new_from_fen("2k3r1/8/8/8/r7/3K4/8/1R5R w - - 0 1")
        self.old_board = GameState()
        self.old_board.board = [
            ["--", "--", "bK", "--", "--", "--", "bR", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["bR", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "wK", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "wR", "--", "--", "--", "--", "--", "wR"],
        ]
        self.old_board.whiteToMove = True

    def test_king_move_generation(self):
        old_generation = self.run_operation(self.run_old_king_generation)
        new_generation = self.run_operation(self.run_new_king_generation)

        print(f'old time: {old_generation}')
        print(f'new time: {new_generation}')

    def run_new_king_generation(self):
        king = self.new_board.current_king()
        king.valid_moves(self.new_board, king.position)

    def run_old_king_generation(self):
        self.old_board.get_king_moves(5, 3, [])

    def run_operation(self, operation):
        start_time = time.time()

        for _ in range(1000):
            operation()

        return time.time() - start_time


if __name__ == '__main__':
    unittest.main()
