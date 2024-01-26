import unittest

import numpy as np

from Chess.board import Board
from Chess.move import Move


# generating different FEN strings is really useful with this webpage:
# https://www.dailychess.com/chess/chess-fen-viewer.php
class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_pawn_move_single(self):
        for i in range(8):
            white_move = Move.from_clicks((6, i), (5, i), self.board)
            black_move = Move.from_clicks((1, i), (2, i), self.board)

            white_move_made = self.board.make_move(white_move)
            black_move_made = self.board.make_move(black_move)

            self.assertTrue(white_move_made)
            self.assertTrue(black_move_made)

        fen = self.board.current_fen()

        self.assertEqual("rnbqkbnr/8/pppppppp/8/8/PPPPPPPP/8/RNBQKBNR w - - 0 1", fen)

    def test_pawn_move_double(self):
        for i in range(8):
            white_move = Move.from_clicks((6, i), (4, i), self.board)
            black_move = Move.from_clicks((1, i), (3, i), self.board)

            white_move_made = self.board.make_move(white_move)
            black_move_made = self.board.make_move(black_move)

            self.assertTrue(white_move_made)
            self.assertTrue(black_move_made)

        fen = self.board.current_fen()

        self.assertEqual("rnbqkbnr/8/8/pppppppp/PPPPPPPP/8/8/RNBQKBNR w - - 0 1", fen)

    def test_white_pawn_capture(self):
        self.board = Board.new_from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w - e6 0 1")

        move = Move.from_clicks((4, 3), (3, 4), self.board)
        move_made = self.board.make_move(move)

        self.assertTrue(move_made)

        fen = self.board.current_fen()

        self.assertEqual("rnbqkbnr/pppp1ppp/8/4P3/8/8/PPP1PPPP/RNBQKBNR b - - 0 1", fen)

    def test_black_pawn_capture(self):
        self.board = Board.new_from_fen("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b - e6 0 1")

        move = Move.from_clicks((3, 4), (4, 3), self.board)
        move_made = self.board.make_move(move)

        self.assertTrue(move_made)

        fen = self.board.current_fen()

        self.assertEqual("rnbqkbnr/pppp1ppp/8/8/3p4/8/PPP1PPPP/RNBQKBNR w - - 0 1", fen)

    def test_white_pawns_cannot_jump(self):
        self.board = Board.new_from_fen("rnbqkbnr/pppppppp/8/8/8/pppppppp/PPPPPPPP/RNBQKBNR w - - 0 1")
        move = Move.from_clicks((6, 0), (4, 0), self.board)
        move_made = self.board.make_move(move)
        self.assertFalse(move_made)

    def test_black_pawns_cannot_jump(self):
        self.board = Board.new_from_fen("rnbqkbnr/pppppppp/PPPPPPPP/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1")
        move = Move.from_clicks((1, 0), (3, 0), self.board)
        move_made = self.board.make_move(move)
        self.assertFalse(move_made)

    def test_white_bishop_moves_up_left_diag(self):
        expected_fens = ["8/8/8/8/8/8/6B1/8 b - - 0 1",
                         "8/8/8/8/8/5B2/8/8 b - - 0 1",
                         "8/8/8/8/4B3/8/8/8 b - - 0 1",
                         "8/8/8/3B4/8/8/8/8 b - - 0 1",
                         "8/8/2B5/8/8/8/8/8 b - - 0 1",
                         "8/1B6/8/8/8/8/8/8 b - - 0 1",
                         "B7/8/8/8/8/8/8/8 b - - 0 1",
                         ]
        for i in range(7):
            self.board = Board.new_from_fen("8/8/8/8/8/8/8/7B w - - 0 1")
            move = Move.from_clicks((7, 7),
                                    (7 - (i + 1), 7 - (i + 1)),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_white_bishop_moves_up_right_diag(self):
        expected_fens = ["8/8/8/8/8/8/1B6/8 b - - 0 1",
                         "8/8/8/8/8/2B5/8/8 b - - 0 1",
                         "8/8/8/8/3B4/8/8/8 b - - 0 1",
                         "8/8/8/4B3/8/8/8/8 b - - 0 1",
                         "8/8/5B2/8/8/8/8/8 b - - 0 1",
                         "8/6B1/8/8/8/8/8/8 b - - 0 1",
                         "7B/8/8/8/8/8/8/8 b - - 0 1",
                         ]
        for i in range(7):
            self.board = Board.new_from_fen("8/8/8/8/8/8/8/B7 w - - 0 1")
            move = Move.from_clicks((7, 0),
                                    (7 - (i + 1), 1 + i),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_white_bishop_moves_down_left_diag(self):
        expected_fens = ["8/1B6/8/8/8/8/8/8 b - - 0 1",
                         "8/8/2B5/8/8/8/8/8 b - - 0 1",
                         "8/8/8/3B4/8/8/8/8 b - - 0 1",
                         "8/8/8/8/4B3/8/8/8 b - - 0 1",
                         "8/8/8/8/8/5B2/8/8 b - - 0 1",
                         "8/8/8/8/8/8/6B1/8 b - - 0 1",
                         "8/8/8/8/8/8/8/7B b - - 0 1"]
        for i in range(7):
            self.board = Board.new_from_fen("B7/8/8/8/8/8/8/8 w - - 0 1")
            move = Move.from_clicks((0, 0),
                                    (i + 1, i + 1),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_white_bishop_moves_down_right_diag(self):
        expected_fens = ["8/6B1/8/8/8/8/8/8 b - - 0 1",
                         "8/8/5B2/8/8/8/8/8 b - - 0 1",
                         "8/8/8/4B3/8/8/8/8 b - - 0 1",
                         "8/8/8/8/3B4/8/8/8 b - - 0 1",
                         "8/8/8/8/8/2B5/8/8 b - - 0 1",
                         "8/8/8/8/8/8/1B6/8 b - - 0 1",
                         "8/8/8/8/8/8/8/B7 b - - 0 1"]
        for i in range(7):
            self.board = Board.new_from_fen("7B/8/8/8/8/8/8/8 w - - 0 1")
            move = Move.from_clicks((0, 7),
                                    (i + 1, 7 - (i + 1)),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_white_bishop_cannot_jump(self):
        self.board = Board.new_from_fen("8/8/2P1P3/3B4/2P1P3/8/8/k5K1 w - - 0 1")
        directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        magnitudes = np.arange(1, 3)
        start = np.array([3, 3])

        for direction in directions:
            for magnitude in magnitudes:
                end = np.add(start, np.multiply(direction, magnitude))
                move = Move(start, end, direction, self.board)

                move_made = self.board.make_move(move)
                self.assertFalse(move_made)
                self.assertEqual("8/8/2P1P3/3B4/2P1P3/8/8/k5K1 w - - 0 1", self.board.current_fen())

    def test_white_bishop_capture(self):
        expected_fens = ["8/8/2B1p3/8/2p1p3/8/8/k5K1 b - - 0 1",
                         "8/8/2p1B3/8/2p1p3/8/8/k5K1 b - - 0 1",
                         "8/8/2p1p3/8/2p1B3/8/8/k5K1 b - - 0 1",
                         "8/8/2p1p3/8/2B1p3/8/8/k5K1 b - - 0 1"]

        directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        start = np.array([3, 3])

        for direction in directions:
            self.board = Board.new_from_fen("8/8/2p1p3/3B4/2p1p3/8/8/k5K1 w - - 0 1")

            end = np.add(start, direction)
            move = Move(start, end, direction, self.board)

            move_made = self.board.make_move(move)
            self.assertTrue(move_made)
            self.assertIn(self.board.current_fen(), expected_fens)

    def test_black_bishop_moves_up_left_diag(self):
        expected_fens = ["8/8/8/8/8/8/6b1/8 w - - 0 1",
                         "8/8/8/8/8/5b2/8/8 w - - 0 1",
                         "8/8/8/8/4b3/8/8/8 w - - 0 1",
                         "8/8/8/3b4/8/8/8/8 w - - 0 1",
                         "8/8/2b5/8/8/8/8/8 w - - 0 1",
                         "8/1b6/8/8/8/8/8/8 w - - 0 1",
                         "b7/8/8/8/8/8/8/8 w - - 0 1",
                         ]
        for i in range(7):
            self.board = Board.new_from_fen("8/8/8/8/8/8/8/7b b - - 0 1")
            move = Move.from_clicks((7, 7),
                                    (7 - (i + 1), 7 - (i + 1)),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_black_bishop_moves_up_right_diag(self):
        expected_fens = ["8/8/8/8/8/8/1b6/8 w - - 0 1",
                         "8/8/8/8/8/2b5/8/8 w - - 0 1",
                         "8/8/8/8/3b4/8/8/8 w - - 0 1",
                         "8/8/8/4b3/8/8/8/8 w - - 0 1",
                         "8/8/5b2/8/8/8/8/8 w - - 0 1",
                         "8/6b1/8/8/8/8/8/8 w - - 0 1",
                         "7b/8/8/8/8/8/8/8 w - - 0 1",
                         ]
        for i in range(7):
            self.board = Board.new_from_fen("8/8/8/8/8/8/8/b7 b - - 0 1")
            move = Move.from_clicks((7, 0),
                                    (7 - (i + 1), 1 + i),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_black_bishop_moves_down_left_diag(self):
        expected_fens = ["8/1b6/8/8/8/8/8/8 w - - 0 1",
                         "8/8/2b5/8/8/8/8/8 w - - 0 1",
                         "8/8/8/3b4/8/8/8/8 w - - 0 1",
                         "8/8/8/8/4b3/8/8/8 w - - 0 1",
                         "8/8/8/8/8/5b2/8/8 w - - 0 1",
                         "8/8/8/8/8/8/6b1/8 w - - 0 1",
                         "8/8/8/8/8/8/8/7b w - - 0 1"]
        for i in range(7):
            self.board = Board.new_from_fen("b7/8/8/8/8/8/8/8 b - - 0 1")
            move = Move.from_clicks((0, 0),
                                    (i + 1, i + 1),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_black_bishop_moves_down_right_diag(self):
        expected_fens = ["8/6b1/8/8/8/8/8/8 w - - 0 1",
                         "8/8/5b2/8/8/8/8/8 w - - 0 1",
                         "8/8/8/4b3/8/8/8/8 w - - 0 1",
                         "8/8/8/8/3b4/8/8/8 w - - 0 1",
                         "8/8/8/8/8/2b5/8/8 w - - 0 1",
                         "8/8/8/8/8/8/1b6/8 w - - 0 1",
                         "8/8/8/8/8/8/8/b7 w - - 0 1"]
        for i in range(7):
            self.board = Board.new_from_fen("7b/8/8/8/8/8/8/8 b - - 0 1")
            move = Move.from_clicks((0, 7),
                                    (i + 1, 7 - (i + 1)),
                                    self.board)
            move_made = self.board.make_move(move)
            self.assertTrue(move_made)

            fen = self.board.current_fen()
            self.assertEqual(expected_fens[i], fen)

    def test_black_bishop_cannot_jump(self):
        self.board = Board.new_from_fen("8/8/2p1p3/3b4/2p1p3/8/8/k5K1 b - - 0 1")
        directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        magnitudes = np.arange(1, 3)
        start = np.array([3, 3])

        for direction in directions:
            for magnitude in magnitudes:
                end = np.add(start, np.multiply(direction, magnitude))
                move = Move(start, end, direction, self.board)

                move_made = self.board.make_move(move)
                self.assertFalse(move_made)
                self.assertEqual("8/8/2p1p3/3b4/2p1p3/8/8/k5K1 b - - 0 1", self.board.current_fen())

    def test_black_bishop_capture(self):
        expected_fens = ["8/8/2b1P3/8/2P1P3/8/8/k5K1 w - - 0 1",
                         "8/8/2P1b3/8/2P1P3/8/8/k5K1 w - - 0 1",
                         "8/8/2P1P3/8/2P1b3/8/8/k5K1 w - - 0 1",
                         "8/8/2P1P3/8/2b1P3/8/8/k5K1 w - - 0 1"]

        directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        start = np.array([3, 3])

        for direction in directions:
            self.board = Board.new_from_fen("8/8/2P1P3/3b4/2P1P3/8/8/k5K1 b - - 0 1")

            end = np.add(start, direction)
            move = Move(start, end, direction, self.board)

            move_made = self.board.make_move(move)
            self.assertTrue(move_made)
            self.assertIn(self.board.current_fen(), expected_fens)


if __name__ == '__main__':
    unittest.main()
