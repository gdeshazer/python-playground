import unittest

import numpy as np

from Chess.board import Board
from Chess.pieces import Bishop, King, Queen, Rook, Pawn, Knight, Empty


class TestChessPieces(unittest.TestCase):

    def setUp(self):
        self.board = None
        self.king = King('w', np.array([3, 3]))
        self.queen = Queen('w')
        self.rook = Rook('w')
        self.pawn = Pawn('w')
        self.knight = Knight('w')
        self.empty = Empty()

    def test_bishop_can_attack(self):
        bishop = Bishop('w')
        current_position = np.array([4, 3])
        targets = np.array([
            [3, 2],
            [2, 1],
            [1, 0],
            [5, 2],
            [6, 1],
            [7, 0],
            [5, 4],
            [6, 5],
            [7, 6],
            [3, 4],
            [2, 5],
            [1, 6],
            [0, 7]
        ])

        for target in targets:
            self.board = Board.new_from_fen('3k4/8/8/8/3B4/8/8/3K4 w - - 0 1')
            self.assertTrue(bishop.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_king_can_attack(self):
        king = King('w', [6, 3])
        current_position = np.array([6, 3])
        targets = np.array([
            [5, 3],
            [5, 2],
            [6, 2],
            [7, 2],
            [7, 3],
            [7, 4],
            [6, 4],
            [5, 4]
        ])

        for target in targets:
            self.board = Board.new_from_fen('3k4/8/8/8/8/8/3K4/8 w - - 0 1')
            self.assertTrue(king.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_queen_can_attack(self):
        queen = Queen('w')
        current_position = np.array([4, 3])
        targets = np.array([
            [3, 2],
            [2, 1],
            [1, 0],
            [5, 2],
            [6, 1],
            [7, 0],
            [5, 4],
            [6, 5],
            [7, 6],
            [3, 4],
            [2, 5],
            [1, 6],
            [0, 7],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
            [5, 3],
            [6, 3],
            [7, 3],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 4],
            [4, 5],
            [4, 6],
            [4, 7]
        ])

        for target in targets:
            self.board = Board.new_from_fen('4k3/8/8/8/3Q4/8/8/2K5 w - - 0 1')
            self.assertTrue(queen.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_rook_can_attack(self):
        rook = Rook('w')
        current_position = np.array([4, 3])
        targets = np.array([
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
            [5, 3],
            [6, 3],
            [7, 3],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 4],
            [4, 5],
            [4, 6],
            [4, 7]
        ])

        for target in targets:
            self.board = Board.new_from_fen('4k3/8/8/8/3R4/8/8/2K5 w - - 0 1')
            self.assertTrue(rook.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_pawn_can_attack(self):
        pawn = Pawn('w')
        current_position = np.array([4, 3])
        targets = np.array([
            [3, 2], [3, 4]
        ])

        for target in targets:
            self.board = Board.new_from_fen('4k3/8/8/8/3P4/8/8/2K5 w - - 0 1')
            self.assertTrue(pawn.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_knight_can_attack(self):
        knight = Knight('w')
        current_position = np.array([4, 3])
        targets = np.array([
            [2, 2],
            [3, 1],
            [5, 1],
            [6, 2],
            [6, 4],
            [5, 5],
            [3, 5],
            [2, 4]
        ])

        for target in targets:
            self.board = Board.new_from_fen('4k3/8/8/8/3N4/8/8/2K5 w - - 0 1')
            self.assertTrue(knight.can_attack_square(current_position, target, self.board),
                            f"couldn't attack {target}")

    def test_empty_cannot_attack(self):
        current_position = np.array([4, 4])
        target = np.array([6, 6])
        self.assertFalse(self.empty.can_attack_square(current_position, target, self.board))


if __name__ == '__main__':
    unittest.main()
