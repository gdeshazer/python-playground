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

    def test_several_sequential_moves(self):
        moves = [
            ((6, 3), (4, 3)),
            ((1, 3), (3, 3)),
            ((7, 1), (5, 2)),
            ((0, 2), (4, 6)),
            ((6, 7), (5, 7))
        ]

        for move in moves:
            move_made = self.board.make_move(Move.from_clicks(move[0], move[1], self.board))
            self.assertTrue(move_made)

        self.assertEqual("rn1qkbnr/ppp1pppp/8/3p4/3P2b1/2N4P/PPP1PPP1/R1BQKBNR b - - 0 1", self.board.current_fen())

    def test_white_bishop_moves_up_left_diag(self):
        expected_fens = ["8/8/8/8/8/8/6B1/8 b - - 0 1",
                         "8/8/8/8/8/5B2/8/8 b - - 0 1",
                         "8/8/8/8/4B3/8/8/8 b - - 0 1",
                         "8/8/8/3B4/8/8/8/8 b - - 0 1",
                         "8/8/2B5/8/8/8/8/8 b - - 0 1",
                         "8/1B6/8/8/8/8/8/8 b - - 0 1",
                         "B7/8/8/8/8/8/8/8 b - - 0 1",
                         ]
        directions = np.array([[-1, -1]])
        magnitudes = np.arange(1, 8)
        start = np.array([7, 7])
        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/8/8/7B w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_white_bishop_moves_up_right_diag(self):
        expected_fens = ["8/8/8/8/8/8/1B6/8 b - - 0 1",
                         "8/8/8/8/8/2B5/8/8 b - - 0 1",
                         "8/8/8/8/3B4/8/8/8 b - - 0 1",
                         "8/8/8/4B3/8/8/8/8 b - - 0 1",
                         "8/8/5B2/8/8/8/8/8 b - - 0 1",
                         "8/6B1/8/8/8/8/8/8 b - - 0 1",
                         "7B/8/8/8/8/8/8/8 b - - 0 1",
                         ]

        directions = np.array([[-1, 1]])
        magnitudes = np.arange(1, 8)
        start = np.array([7, 0])
        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/8/8/B7 w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_white_bishop_moves_down_left_diag(self):
        expected_fens = ["8/1B6/8/8/8/8/8/8 b - - 0 1",
                         "8/8/2B5/8/8/8/8/8 b - - 0 1",
                         "8/8/8/3B4/8/8/8/8 b - - 0 1",
                         "8/8/8/8/4B3/8/8/8 b - - 0 1",
                         "8/8/8/8/8/5B2/8/8 b - - 0 1",
                         "8/8/8/8/8/8/6B1/8 b - - 0 1",
                         "8/8/8/8/8/8/8/7B b - - 0 1"]

        directions = np.array([[1, 1]])
        magnitudes = np.arange(1, 8)
        start = np.array([0, 0])
        self.move_and_validate_directions_with_magnitude("B7/8/8/8/8/8/8/8 w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_white_bishop_moves_down_right_diag(self):
        expected_fens = ["8/6B1/8/8/8/8/8/8 b - - 0 1",
                         "8/8/5B2/8/8/8/8/8 b - - 0 1",
                         "8/8/8/4B3/8/8/8/8 b - - 0 1",
                         "8/8/8/8/3B4/8/8/8 b - - 0 1",
                         "8/8/8/8/8/2B5/8/8 b - - 0 1",
                         "8/8/8/8/8/8/1B6/8 b - - 0 1",
                         "8/8/8/8/8/8/8/B7 b - - 0 1"]

        directions = np.array([[1, -1]])
        magnitudes = np.arange(1, 8)
        start = np.array([0, 7])
        self.move_and_validate_directions_with_magnitude("7B/8/8/8/8/8/8/8 w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

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

        directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        start = np.array([3, 3])

        self.move_and_validate_directions("8/8/2p1p3/3B4/2p1p3/8/8/k5K1 w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_black_bishop_moves_up_left_diag(self):
        expected_fens = ["8/8/8/8/8/8/6b1/8 w - - 0 1",
                         "8/8/8/8/8/5b2/8/8 w - - 0 1",
                         "8/8/8/8/4b3/8/8/8 w - - 0 1",
                         "8/8/8/3b4/8/8/8/8 w - - 0 1",
                         "8/8/2b5/8/8/8/8/8 w - - 0 1",
                         "8/1b6/8/8/8/8/8/8 w - - 0 1",
                         "b7/8/8/8/8/8/8/8 w - - 0 1",
                         ]

        directions = np.array([[-1, -1]])
        magnitudes = np.arange(1, 8)
        start = np.array([7, 7])
        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/8/8/7b b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_black_bishop_moves_up_right_diag(self):
        expected_fens = ["8/8/8/8/8/8/1b6/8 w - - 0 1",
                         "8/8/8/8/8/2b5/8/8 w - - 0 1",
                         "8/8/8/8/3b4/8/8/8 w - - 0 1",
                         "8/8/8/4b3/8/8/8/8 w - - 0 1",
                         "8/8/5b2/8/8/8/8/8 w - - 0 1",
                         "8/6b1/8/8/8/8/8/8 w - - 0 1",
                         "7b/8/8/8/8/8/8/8 w - - 0 1",
                         ]

        directions = np.array([[-1, 1]])
        magnitudes = np.arange(1, 8)
        start = np.array([7, 0])
        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/8/8/b7 b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_black_bishop_moves_down_left_diag(self):
        expected_fens = ["8/1b6/8/8/8/8/8/8 w - - 0 1",
                         "8/8/2b5/8/8/8/8/8 w - - 0 1",
                         "8/8/8/3b4/8/8/8/8 w - - 0 1",
                         "8/8/8/8/4b3/8/8/8 w - - 0 1",
                         "8/8/8/8/8/5b2/8/8 w - - 0 1",
                         "8/8/8/8/8/8/6b1/8 w - - 0 1",
                         "8/8/8/8/8/8/8/7b w - - 0 1"]

        directions = np.array([[1, 1]])
        magnitudes = np.arange(1, 8)
        start = np.array([0, 0])
        self.move_and_validate_directions_with_magnitude("b7/8/8/8/8/8/8/8 b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_black_bishop_moves_down_right_diag(self):
        expected_fens = ["8/6b1/8/8/8/8/8/8 w - - 0 1",
                         "8/8/5b2/8/8/8/8/8 w - - 0 1",
                         "8/8/8/4b3/8/8/8/8 w - - 0 1",
                         "8/8/8/8/3b4/8/8/8 w - - 0 1",
                         "8/8/8/8/8/2b5/8/8 w - - 0 1",
                         "8/8/8/8/8/8/1b6/8 w - - 0 1",
                         "8/8/8/8/8/8/8/b7 w - - 0 1"]

        directions = np.array([[1, -1]])
        magnitudes = np.arange(1, 8)
        start = np.array([0, 7])
        self.move_and_validate_directions_with_magnitude("7b/8/8/8/8/8/8/8 b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

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

        self.move_and_validate_directions("8/8/2P1P3/3b4/2P1P3/8/8/k5K1 b - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_knight_moves(self):
        expected_fens = ["8/2N5/8/8/8/8/8/K1k5 b - - 0 1",
                         "8/8/1N6/8/8/8/8/K1k5 b - - 0 1",
                         "8/8/8/8/1N6/8/8/K1k5 b - - 0 1",
                         "8/8/8/8/8/2N5/8/K1k5 b - - 0 1",
                         "8/8/8/8/8/4N3/8/K1k5 b - - 0 1",
                         "8/8/8/8/5N2/8/8/K1k5 b - - 0 1",
                         "8/8/5N2/8/8/8/8/K1k5 b - - 0 1",
                         "8/4N3/8/8/8/8/8/K1k5 b - - 0 1"
                         ]
        directions = np.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], np.int8)

        start = np.array([3, 3])
        self.move_and_validate_directions("8/8/8/3N4/8/8/8/K1k5 w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_knight_can_jump(self):
        expected_fens = ["8/2N5/2pPP3/2p1p3/2PPp3/8/8/K1k5 b - - 0 1",
                         "8/8/1NpPP3/2p1p3/2PPp3/8/8/K1k5 b - - 0 1",
                         "8/8/2pPP3/2p1p3/1NPPp3/8/8/K1k5 b - - 0 1",
                         "8/8/2pPP3/2p1p3/2PPp3/2N5/8/K1k5 b - - 0 1",
                         "8/8/2pPP3/2p1p3/2PPp3/4N3/8/K1k5 b - - 0 1",
                         "8/8/2pPP3/2p1p3/2PPpN2/8/8/K1k5 b - - 0 1",
                         "8/8/2pPPN2/2p1p3/2PPp3/8/8/K1k5 b - - 0 1",
                         "8/4N3/2pPP3/2p1p3/2PPp3/8/8/K1k5 b - - 0 1"
                         ]
        directions = np.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], np.int8)

        start = np.array([3, 3])
        self.move_and_validate_directions("8/8/2pPP3/2pNp3/2PPp3/8/8/K1k5 w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_white_knight_can_capture(self):
        expected_fens = ["8/2N1p3/1qpPPp2/2p1n3/1rPPpn2/2r1b3/8/K6k b - - 0 1",
                         "8/2p1p3/1NpPPp2/2p1n3/1rPPpn2/2r1b3/8/K6k b - - 0 1",
                         "8/2p1p3/1qpPPp2/2p1n3/1NPPpn2/2r1b3/8/K6k b - - 0 1",
                         "8/2p1p3/1qpPPp2/2p1n3/1rPPpn2/2N1b3/8/K6k b - - 0 1",
                         "8/2p1p3/1qpPPp2/2p1n3/1rPPpn2/2r1N3/8/K6k b - - 0 1",
                         "8/2p1p3/1qpPPp2/2p1n3/1rPPpN2/2r1b3/8/K6k b - - 0 1",
                         "8/2p1p3/1qpPPN2/2p1n3/1rPPpn2/2r1b3/8/K6k b - - 0 1",
                         "8/2p1N3/1qpPPp2/2p1n3/1rPPpn2/2r1b3/8/K6k b - - 0 1"
                         ]
        directions = np.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], np.int8)

        start = np.array([3, 3])
        self.move_and_validate_directions("8/2p1p3/1qpPPp2/2pNn3/1rPPpn2/2r1b3/8/K6k w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_black_knight_can_capture(self):
        expected_fens = ["8/2n1P3/1QpPPP2/2p1n3/1RPPpN2/2R1B3/8/K6k w - - 0 1",
                         "8/2P1P3/1npPPP2/2p1n3/1RPPpN2/2R1B3/8/K6k w - - 0 1",
                         "8/2P1P3/1QpPPP2/2p1n3/1nPPpN2/2R1B3/8/K6k w - - 0 1",
                         "8/2P1P3/1QpPPP2/2p1n3/1RPPpN2/2n1B3/8/K6k w - - 0 1",
                         "8/2P1P3/1QpPPP2/2p1n3/1RPPpN2/2R1n3/8/K6k w - - 0 1",
                         "8/2P1P3/1QpPPP2/2p1n3/1RPPpn2/2R1B3/8/K6k w - - 0 1",
                         "8/2P1P3/1QpPPn2/2p1n3/1RPPpN2/2R1B3/8/K6k w - - 0 1",
                         "8/2P1n3/1QpPPP2/2p1n3/1RPPpN2/2R1B3/8/K6k w - - 0 1"
                         ]
        directions = np.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], np.int8)

        start = np.array([3, 3])
        self.move_and_validate_directions("8/2P1P3/1QpPPP2/2pnn3/1RPPpN2/2R1B3/8/K6k b - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_white_knight_cannot_capture_friendly(self):
        directions = np.array([[2, 1], [1, 2]])

        start = np.array([3, 2])
        self.board = Board.new_from_fen("8/8/8/2N2n2/3pP3/3Pp3/8/K6k w - - 0 1")
        for direction in directions:
            end = np.add(direction, start)
            move = Move(start, end, direction, self.board)

            move_made = self.board.make_move(move)
            self.assertFalse(move_made)
            self.assertEqual("8/8/8/2N2n2/3pP3/3Pp3/8/K6k w - - 0 1", self.board.current_fen())

    def test_black_knight_cannot_capture_friendly(self):
        directions = np.array([[2, -1], [1, -2]])

        start = np.array([3, 5])
        self.board = Board.new_from_fen("8/8/8/2N2n2/3pP3/3Pp3/8/K6k b - - 0 1")
        for direction in directions:
            end = np.add(direction, start)
            move = Move(start, end, direction, self.board)

            move_made = self.board.make_move(move)
            self.assertFalse(move_made)
            self.assertEqual("8/8/8/2N2n2/3pP3/3Pp3/8/K6k b - - 0 1", self.board.current_fen())

    def test_white_rook_valid_moves(self):
        expected_fens = ["8/8/8/8/8/7k/R4K2/8 b - - 0 1",
                         "8/8/8/8/8/R6k/5K2/8 b - - 0 1",
                         "8/8/8/8/R7/7k/5K2/8 b - - 0 1",
                         "8/8/8/R7/8/7k/5K2/8 b - - 0 1",
                         "8/8/R7/8/8/7k/5K2/8 b - - 0 1",
                         "8/R7/8/8/8/7k/5K2/8 b - - 0 1",
                         "R7/8/8/8/8/7k/5K2/8 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/1R6 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/2R5 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/3R4 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/4R3 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/5R2 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/6R1 b - - 0 1",
                         "8/8/8/8/8/7k/5K2/7R b - - 0 1",
                         "6R1/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "5R2/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "4R3/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "3R4/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "2R5/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "1R6/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "R7/8/8/8/5k2/8/5K2/8 b - - 0 1",
                         "8/7R/8/8/5k2/8/5K2/8 b - - 0 1",
                         "8/8/7R/8/5k2/8/5K2/8 b - - 0 1",
                         "8/8/8/7R/5k2/8/5K2/8 b - - 0 1",
                         "8/8/8/8/5k1R/8/5K2/8 b - - 0 1",
                         "8/8/8/8/5k2/7R/5K2/8 b - - 0 1",
                         "8/8/8/8/5k2/8/5K1R/8 b - - 0 1",
                         "8/8/8/8/5k2/8/5K2/7R b - - 0 1"
                         ]
        # first set of directions moving up, and moving right
        directions = np.array([[-1, 0], [0, 1]], np.int8)
        magnitudes = np.arange(1, 8)
        start = np.array([7, 0])

        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/7k/5K2/R7 w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

        # first set of directions moving down, and moving left
        directions = np.array([[1, 0], [0, -1]], np.int8)
        magnitudes = np.arange(1, 8)
        start = np.array([0, 7])

        self.move_and_validate_directions_with_magnitude("7R/8/8/8/5k2/8/5K2/8 w - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_black_rook_valid_moves(self):
        expected_fens = ["8/8/8/8/8/7k/r4K2/8 w - - 0 1",
                         "8/8/8/8/8/r6k/5K2/8 w - - 0 1",
                         "8/8/8/8/r7/7k/5K2/8 w - - 0 1",
                         "8/8/8/r7/8/7k/5K2/8 w - - 0 1",
                         "8/8/r7/8/8/7k/5K2/8 w - - 0 1",
                         "8/r7/8/8/8/7k/5K2/8 w - - 0 1",
                         "r7/8/8/8/8/7k/5K2/8 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/1r6 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/2r5 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/3r4 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/4r3 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/5r2 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/6r1 w - - 0 1",
                         "8/8/8/8/8/7k/5K2/7r w - - 0 1",
                         "6r1/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "5r2/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "4r3/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "3r4/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "2r5/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "1r6/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "r7/8/8/8/5k2/8/5K2/8 w - - 0 1",
                         "8/7r/8/8/5k2/8/5K2/8 w - - 0 1",
                         "8/8/7r/8/5k2/8/5K2/8 w - - 0 1",
                         "8/8/8/7r/5k2/8/5K2/8 w - - 0 1",
                         "8/8/8/8/5k1r/8/5K2/8 w - - 0 1",
                         "8/8/8/8/5k2/7r/5K2/8 w - - 0 1",
                         "8/8/8/8/5k2/8/5K1r/8 w - - 0 1",
                         "8/8/8/8/5k2/8/5K2/7r w - - 0 1"
                         ]
        # first set of directions moving up, and moving right
        directions = np.array([[-1, 0], [0, 1]], np.int8)
        magnitudes = np.arange(1, 8)
        start = np.array([7, 0])

        self.move_and_validate_directions_with_magnitude("8/8/8/8/8/7k/5K2/r7 b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

        # first set of directions moving down, and moving left
        directions = np.array([[1, 0], [0, -1]], np.int8)
        magnitudes = np.arange(1, 8)
        start = np.array([0, 7])

        self.move_and_validate_directions_with_magnitude("7r/8/8/8/5k2/8/5K2/8 b - - 0 1",
                                                         directions,
                                                         magnitudes,
                                                         expected_fens,
                                                         start)

    def test_white_rook_cannot_hop(self):
        self.board = Board.new_from_fen("8/8/8/3P4/2PRP3/3P3k/6K1/8 w - - 0 1")
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.int8)
        start = np.array([4, 3])
        for direction in directions:
            end = np.add(start, direction)
            move = Move(start, end, direction, self.board)

            self.assertFalse(self.board.make_move(move))
            self.assertEqual("8/8/8/3P4/2PRP3/3P3k/6K1/8 w - - 0 1", self.board.current_fen())

    def test_black_rook_cannot_hop(self):
        self.board = Board.new_from_fen("8/8/8/3p4/2prp3/3p3k/8/5K2 b - - 0 1")
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.int8)
        start = np.array([4, 3])
        for direction in directions:
            end = np.add(start, direction)
            move = Move(start, end, direction, self.board)

            self.assertFalse(self.board.make_move(move))
            self.assertEqual("8/8/8/3p4/2prp3/3p3k/8/5K2 b - - 0 1", self.board.current_fen())

    def test_white_rook_captures(self):
        expected_fens = ["8/8/8/3R4/2p1p3/3p3k/8/5K2 b - - 0 1",
                         "8/8/8/3p4/2R1p3/3p3k/8/5K2 b - - 0 1",
                         "8/8/8/3p4/2p1p3/3R3k/8/5K2 b - - 0 1",
                         "8/8/8/3p4/2p1R3/3p3k/8/5K2 b - - 0 1"
                         ]
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.int8)
        start = np.array([4, 3])
        self.move_and_validate_directions("8/8/8/3p4/2pRp3/3p3k/8/5K2 w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_black_rook_captures(self):
        expected_fens = ["8/8/8/3r4/2P1P3/3P3k/8/5K2 w - - 0 1",
                         "8/8/8/3P4/2r1P3/3P3k/8/5K2 w - - 0 1",
                         "8/8/8/3P4/2P1P3/3r3k/8/5K2 w - - 0 1",
                         "8/8/8/3P4/2P1r3/3P3k/8/5K2 w - - 0 1"
                         ]
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.int8)
        start = np.array([4, 3])
        self.move_and_validate_directions("8/8/8/3P4/2PrP3/3P3k/8/5K2 b - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_queen_moves(self):
        magnitudes = np.arange(1, 8)

        tests = [
            {
                'name': 'up-vertical, right-horizontal, up-right diagonal',
                'start_position': [7, 0],
                'start_fen': "8/8/8/8/8/8/8/Q7 w - - 0 1",
                'directions': np.array([[-1, 0], [0, 1], [-1, 1]], np.int8)
            },
            {
                'name': 'up-vertical, left-horizontal, up-left diagonal',
                'start_position': [7, 7],
                'start_fen': "8/8/8/8/8/8/8/7Q w - - 0 1",
                'directions': np.array([[-1, 0], [0, -1], [-1, -1]], np.int8)
            },
            {
                'name': 'down-vertical, left-horizontal, down-left diagonal',
                'start_position': [0, 7],
                'start_fen': "7Q/8/8/8/8/8/8/8 w - - 0 1",
                'directions': np.array([[1, 0], [0, -1], [1, -1]], np.int8)
            },
            {
                'name': 'down-vertical, right-horizontal, down-right diagonal',
                'start_position': [0, 0],
                'start_fen': "Q7/8/8/8/8/8/8/8 w - - 0 1",
                'directions': np.array([[1, 0], [0, 1], [1, 1]], np.int8)
            }
        ]

        for test in tests:
            with self.subTest(test['name']):
                start = test['start_position']
                directions = test['directions']
                for direction in directions:
                    for magnitude in magnitudes:
                        end = np.add(start, np.multiply(direction, magnitude))
                        expected_fen = self.generate_fen('Q', end, 'b')

                        board = Board.new_from_fen(test['start_fen'])
                        move = Move(start, end, direction, board)

                        self.assertTrue(board.make_move(move))
                        self.assertEqual(expected_fen, board.current_fen())

    def test_queen_captures(self):
        expected_fens = [
            "8/8/8/2nQb3/2p1r3/2pkq3/8/7K b - - 0 1",
            "8/8/8/2Qnb3/2p1r3/2pkq3/8/7K b - - 0 1",
            "8/8/8/2nnb3/2Q1r3/2pkq3/8/7K b - - 0 1",
            "8/8/8/2nnb3/2p1r3/2Qkq3/8/7K b - - 0 1",
            "8/8/8/2nnb3/2p1r3/2pQq3/8/7K b - - 0 1",
            "8/8/8/2nnb3/2p1r3/2pkQ3/8/7K b - - 0 1",
            "8/8/8/2nnb3/2p1Q3/2pkq3/8/7K b - - 0 1",
            "8/8/8/2nnQ3/2p1r3/2pkq3/8/7K b - - 0 1"
        ]

        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        start = [4, 3]
        self.move_and_validate_directions("8/8/8/2nnb3/2pQr3/2pkq3/8/7K w - - 0 1",
                                          directions,
                                          expected_fens,
                                          start)

    def test_king_moves(self):
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
        start = [4, 4]
        for direction in directions:
            self.board = Board.new_from_fen("p7/8/8/8/4K3/8/8/8 w - - 0 1")
            end = np.add(direction, start)

            # the generate fen method could be updated to include an additional piece, but this seems easier for now
            expected_fen = "p7" + self.generate_fen('K', end, 'b')[1:]
            move = Move(start, end, direction, self.board)

            self.assertTrue(self.board.make_move(move))
            self.assertEqual(expected_fen, self.board.current_fen())

    def test_king_captures(self):
        tests = [
            {
                'name': 'up down captures',
                "start_position": [3, 3],
                "start_fen": "8/8/3p4/3K4/3p4/8/8/k7 w - - 0 1",
                "expected_fens": ["8/8/3K4/8/3p4/8/8/k7 b - - 0 1", "8/8/3p4/8/3K4/8/8/k7 b - - 0 1"],
                "directions": np.array([[-1, 0], [1, 0]])
            },
            {
                'name': 'left right captures',
                "start_position": [3, 3],
                "start_fen": "8/8/8/2pKp3/8/8/8/k7 w - - 0 1",
                "expected_fens": ["8/8/8/2p1K3/8/8/8/k7 b - - 0 1", "8/8/8/2K1p3/8/8/8/k7 b - - 0 1"],
                "directions": np.array([[0, -1], [0, 1]])
            },
            {
                'name': 'up diagonal captures',
                "start_position": [3, 3],
                "start_fen": "8/8/2p1p3/3K4/8/8/8/k7 w - - 0 1",
                "expected_fens": ["8/8/2p1K3/8/8/8/8/k7 b - - 0 1", "8/8/2K1p3/8/8/8/8/k7 b - - 0 1"],
                "directions": np.array([[-1, -1], [-1, 1]])
            },
            {
                'name': 'down diagonal captures',
                "start_position": [3, 3],
                "start_fen": "8/8/8/3K4/2p1p3/8/8/k7 w - - 0 1",
                "expected_fens": ["8/8/8/8/2K1p3/8/8/k7 b - - 0 1", "8/8/8/8/2p1K3/8/8/k7 b - - 0 1"],
                "directions": np.array([[1, -1], [1, 1]])
            }
        ]

        for test in tests:
            with self.subTest(test['name']):
                start_position = test["start_position"]
                start_fen = test["start_fen"]
                expected_fens = test["expected_fens"]
                directions = test["directions"]
                self.move_and_validate_directions(start_fen, directions, expected_fens, start_position)

    def test_king_in_check(self):
        test = [
            {
                'name': 'king double checked - only king can move',
                'start_fen': '8/1k6/3r4/8/3K2r1/3B4/8/8 w - - 0 1',
                'piece_to_move': [4, 3],
                'direction': [-1, 1],
                'can_move': True,
                'end_fen': '8/1k6/3r4/4K3/6r1/3B4/8/8 b - - 0 1'
            },
            {
                'name': 'king checked - only king can move',
                'start_fen': '8/1k6/3r4/8/3K4/3B4/8/8 w - - 0 1',
                'piece_to_move': [5, 3],
                'direction': [-1, -1],
                'can_move': False,
                'end_fen': '8/1k6/3r4/8/3K4/3B4/8/8 w - - 0 1'
            },
            {
                'name': 'king checked - can capture with other',
                'start_fen': '8/1k6/3r4/8/3K4/6B1/8/8 w - - 0 1',
                'piece_to_move': [5, 6],
                'direction': [-3, -3],
                'can_move': True,
                'end_fen': '8/1k6/3B4/8/3K4/8/8/8 b - - 0 1'
            },
            {
                'name': 'king checked - can capture with king',
                'start_fen': '8/1k6/8/3r4/3K4/3B4/8/8 w - - 0 1',
                'piece_to_move': [4, 3],
                'direction': [-1, 0],
                'can_move': True,
                'end_fen': '8/1k6/8/3K4/8/3B4/8/8 b - - 0 1'
            },
            {
                'name': 'king checked - can block check',
                'start_fen': '8/1k6/3r4/R7/3K4/3B4/8/8 w - - 0 1',
                'piece_to_move': [3, 0],
                'direction': [0, 3],
                'can_move': True,
                'end_fen': '8/1k6/3r4/3R4/3K4/3B4/8/8 b - - 0 1'
            }
        ]

        for test_case in test:
            with self.subTest(test_case['name']):
                self.board = Board.new_from_fen(test_case['start_fen'])
                start = test_case['piece_to_move']
                direction = np.array(test_case['direction'])
                move = Move(start, np.add(start, direction), direction, self.board)

                can_move = test_case['can_move']
                end_fen = test_case['end_fen']

                self.assertEqual(self.board.make_move(move), can_move)
                self.assertEqual(end_fen, self.board.current_fen())

    def test_cannot_move_king_into_check(self):
        self.board = Board.new_from_fen("k7/8/8/3q4/8/4K3/8/8 w - - 0 1")
        move = Move.from_clicks((5, 4), (5, 5), self.board)
        self.assertFalse(self.board.make_move(move))

    def test_cannot_move_pinned_piece(self):
        self.board = Board.new_from_fen("8/1k6/3r4/8/8/3B4/3K4/8 w - - 0 1")
        move = Move.from_clicks((5, 3), (4, 4), self.board)
        self.assertFalse(self.board.make_move(move))

    def test_pinned_bishop_moves_in_two_directions(self):
        tests = [
            {
                "name": "bishop-captures",
                "start": (2, 2),
                "end": (4, 0),
                "end_fen": "4k3/8/8/8/b7/8/8/4K3 w - - 0 1"
            },
            {
                "name": "bishop-slide-down",
                "start": (2, 2),
                "end": (3, 1),
                "end_fen": "4k3/8/8/1b6/B7/8/8/4K3 w - - 0 1"
            },
            {
                "name": "bishop-slide-up",
                "start": (2, 2),
                "end": (1, 3),
                "end_fen": "4k3/3b4/8/8/B7/8/8/4K3 w - - 0 1"
            },
        ]

        for test in tests:
            with self.subTest(test['name']):
                self.board = Board.new_from_fen("4k3/8/2b5/8/B7/8/8/4K3 b - - 0 1")
                move = Move.from_clicks(test['start'], test['end'], self.board)
                move_was_made = self.board.make_move(move)
                self.assertTrue(move_was_made)
                self.assertEqual(test['end_fen'], self.board.current_fen())

    def test_pinned_pawn_cannot_move(self):
        self.board = Board.new_from_fen("3k4/8/8/b7/8/8/3P4/4K3 w - - 0 1")
        move = Move.from_clicks((6, 3), (5, 3), self.board)
        self.assertFalse(self.board.make_move(move))

    def move_and_validate_directions(self, initial_fen, directions, expected_fens, start):
        for direction in directions:
            self.board = Board.new_from_fen(initial_fen)

            end = np.add(start, direction)
            move = Move(start, end, direction, self.board)

            move_made = self.board.make_move(move)
            self.assertTrue(move_made)
            self.assertIn(self.board.current_fen(), expected_fens)

    def move_and_validate_directions_with_magnitude(self, initial_fen, directions, magnitudes, expected_fens,
                                                    start):
        for direction in directions:
            for magnitude in magnitudes:
                self.board = Board.new_from_fen(initial_fen)

                end = np.add(start, np.multiply(direction, magnitude))
                move = Move(start, end, direction, self.board)

                move_made = self.board.make_move(move)
                self.assertTrue(move_made)
                self.assertIn(self.board.current_fen(), expected_fens)

    def generate_fen(self, piece: str,
                     position,
                     current_turn: str = "w",
                     castling: str = "-",
                     en_passant: str = "-",
                     half_move: int = 0,
                     full_move: int = 1) -> str:

        # Initialize the empty 8x8 board
        board = [["" for _ in range(8)] for _ in range(8)]

        # Add the piece to the board
        board[position[0]][position[1]] = piece

        # Generate the FEN string
        fen = ""
        for row in board:
            empty = 0
            for cell in row:
                if cell == "":
                    empty += 1
                else:
                    if empty > 0:
                        fen += str(empty)
                        empty = 0
                    fen += cell
            if empty > 0:
                fen += str(empty)
            fen += "/"

        # Remove the trailing slash
        fen = fen[:-1]

        # Add the placeholders for the current turn, castling, en passant, half move and full move counters
        fen += f" {current_turn} {castling} {en_passant} {half_move} {full_move}"

        return fen

    if __name__ == '__main__':
        unittest.main()
