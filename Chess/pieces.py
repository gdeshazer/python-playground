from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Piece(ABC):

    def __init__(self, color: str, name: str) -> None:
        self.color = color
        self.name = name
        self.is_pinned = False

    @classmethod
    def from_str(cls, name: str) -> "Piece":
        # convert to upper case just incase
        piece_type = name[1].upper()
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
        elif piece_type == 'P':
            return Pawn(color)
        elif piece_type == "-":
            return Empty()
        else:
            return Empty()

    @abstractmethod
    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        pass

    def position_id(self, position: Union[np.ndarray, list[int]]) -> str:
        return f'{self.color}{self.name}-{position[0]}{position[1]}'

    # todo: there should be a better way to handle this to, maybe the piece should store it's current position?
    def id_to_position(self, id_str: str) -> np.ndarray[np.int8]:
        # noinspection PyTypeChecker
        return np.array([int(id_str[-2]), int(id_str[-1])], dtype=np.int8)

    def full_name(self) -> str:
        return f'{self.color}{self.name}'

    def get_chess_notation(self) -> str:
        if self.color == 'w':
            return self.name.upper()
        else:
            return self.name.lower()

    def build_moves_from_directions(self,
                                    start: np.ndarray,
                                    directions: np.ndarray,
                                    magnitudes: np.ndarray,
                                    board) -> list:
        """
        create a list of moves based on a starting position, a set of direction vectors ie: [ [1,0], [0,1] ] and a
        set of magnitudes, ie: [1, 2, 3, 4] (basically how far to move in a given direction)
        :return: list of possible moves for all magnitudes and directions
        """
        from Chess.move import Move
        moves = []
        enemy_color = 'b' if self.color == 'w' else 'w'

        for direction in directions:
            for magnitude in magnitudes:
                dir_vector = np.multiply(direction, magnitude)
                move_vector = np.add(start, dir_vector)

                # check out of bounds
                if np.any(move_vector > 7) or np.any(move_vector < 0):
                    break

                target = board.piece_at(move_vector)

                if isinstance(target, Empty):
                    moves.append(Move(start, move_vector, direction, board))
                    continue
                elif target.color == enemy_color:
                    # we don't need to look any further in this direction
                    moves.append(Move(start, move_vector, direction, board))
                    break
                else:
                    # we are blocked by one of our own pieces so no need to look further in this direction
                    break

        return moves

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
        super().__init__(color, 'P')

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
        from Chess.move import Move
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

                    # pawns in their starting row can move two spaces forward, but they can't hop anything
                    # so if we didn't add a move for the pawn to move forward once, we can't add one to move twice
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
    DIRECTIONS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'B')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)


class Knight(Piece):
    DIRECTIONS = np.array([
        [2, 1],
        [2, -1],
        [-2, 1],
        [-2, -1],
        [1, 2],
        [1, -2],
        [-1, 2],
        [-1, -2]
    ], np.int8)
    MAGNITUDES = np.array([1])

    def __init__(self, color: str) -> None:
        super().__init__(color, 'N')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)


class Rook(Piece):
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.int8)
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'R')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)


class Queen(Piece):
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'Q')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)


class King(Piece):
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
    MAGNITUDES = np.array([1])

    def __init__(self, color: str) -> None:
        super().__init__(color, 'K')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        # todo: need to ensure that the move being made does not place the king in check
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)
