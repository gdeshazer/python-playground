from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Piece(ABC):

    def __init__(self, color: str, name: str) -> None:
        self.color = color
        self.name = name
        self.isPinned = False

    @classmethod
    def from_str(cls, name: str) -> "Piece":
        piece_type = name[1]
        color = name[0]

        if piece_type == 'R':
            return Rook(color)
        elif piece_type == 'N':
            return Knight(color)
        elif piece_type == 'B':
            return Bishop(color)
        elif piece_type == 'Q':
            return Queen(color)
        elif piece_type == 'K':
            return King(color)
        elif piece_type == 'p':
            return Pawn(color)
        elif piece_type == "-":
            return Empty()
        else:
            return Empty()

    @abstractmethod
    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        pass

    def make_move(self, board: list[list[str]], move) -> None:
        # if move.cl
        pass

    def position_id(self, position: Union[np.ndarray, list[int]]) -> str:
        return f'{self.color}{self.name}-{position[0]}{position[1]}'

    # todo: there should be a better way to handle this to, maybe the piece should store it's current position?
    def id_to_position(self, id_str: str) -> np.ndarray[np.int8]:
        # noinspection PyTypeChecker
        return np.array([int(id_str[-2]), int(id_str[-1])], dtype=np.int8)


    def full_name(self) -> str:
        return f'{self.color}{self.name}'

    def __str__(self):
        return self.full_name()


class Empty(Piece):
    def __init__(self) -> None:
        super().__init__('-', '-')

    def valid_moves(self, board: list[list["Piece"]], position: np.ndarray) -> list:
        return []


class Pawn(Piece):
    # noinspection PyTypeChecker
    WHITE_DIRECTIONS: np.ndarray[np.int8] = np.array([[-1, 0], [-1, 1], [-1, -1]], np.int8)

    # noinspection PyTypeChecker
    BLACK_DIRECTIONS: np.ndarray[np.int8] = np.array([[1, 0], [1, 1], [1, -1]], np.int8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'p')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        if self.color == 'w':
            return self.calculate_pawn_moves(board, position, self.WHITE_DIRECTIONS, position[0] == 6, 'b')
        elif self.color == 'b':
            return self.calculate_pawn_moves(board, position, self.BLACK_DIRECTIONS, position[0] == 1, 'w')

    def calculate_pawn_moves(self,
                             board,
                             start: np.ndarray[np.int8],
                             directions: np.ndarray,
                             is_in_start: bool,
                             capture_color: str) -> list:
        from move import Move
        moves: list = []

        for direction in directions:
            direction_product = np.prod(direction)
            endpoint = np.add(start, direction)

            # we are outside the board
            if np.any(endpoint > 7) or np.any(endpoint < 0):
                break

            target = board.piece_at([endpoint[0], endpoint[1]])

            # we are moving vertically
            if direction_product == 0:
                if isinstance(target, Empty):
                    moves.append(Move(start, endpoint, direction, board))

                # pawns in their starting row can move two spaces forward
                if is_in_start:
                    additional_move = np.add(endpoint, direction)
                    new_target = board.piece_at([additional_move[0], additional_move[1]])
                    if isinstance(new_target, Empty):
                        moves.append(Move(start, additional_move, direction, board))

            # moving in diagonals to capture
            elif target.color == capture_color:
                moves.append(Move(start, endpoint, direction, board))

        return moves


class Bishop(Piece):
    def __init__(self, color: str) -> None:
        super().__init__(color, 'B')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return []


class Knight(Piece):
    def __init__(self, color: str) -> None:
        super().__init__(color, 'K')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return []


class Rook(Piece):
    def __init__(self, color: str) -> None:
        super().__init__(color, 'R')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return []


class Queen(Piece):
    def __init__(self, color: str) -> None:
        super().__init__(color, 'Q')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return []


class King(Piece):
    def __init__(self, color: str) -> None:
        super().__init__(color, 'K')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return []
