import unittest

from Chess.board import Board
from Chess.move import Move


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_white_pawn_move_single(self):
        move = Move.from_clicks((6, 0), (5, 0), self.board)

        move_made = self.board.make_move(move)

        self.assertTrue(move_made)

        # After move string should be different as position has changed
        fen = self.board.current_fen()

        self.assertEqual(fen, "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1")


if __name__ == '__main__':
    unittest.main()
