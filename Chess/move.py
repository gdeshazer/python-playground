from typing import Union

import numpy as np

ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
fields_to_col = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
col_to_field = {v: k for k, v in fields_to_col.items()}


def get_rank_file(position) -> str:
    return col_to_field[position[1]] + rows_to_ranks[position[0]]


class Move:
    """
    The move class is responsible for containing information about a possible move
    """

    # for some reason the type checker won't allow for the list to be fully coerced into the numpy types
    # noinspection PyTypeChecker
    def __init__(self,
                 start_square: np.ndarray[np.int8],
                 end_square: np.ndarray[np.int8],
                 direction: np.ndarray[np.int8],
                 board) -> None:
        from Chess.pieces import Piece
        self.start_position: np.ndarray[np.int8] = start_square
        self.end_position: np.ndarray[np.int8] = end_square
        self.direction: np.ndarray[np.int8] = direction
        self.en_passant: Union[None, np.ndarray[np.int8]] = None
        self.is_promotion: bool = False

        # while it would be better to import the board type here, we get a circular import warning which python
        # can't figure out.  once the new board's logic is validated to be functional, we might be able to switch back
        # to using isinstance(board, Board)
        if callable(getattr(board, 'piece_at', None)):
            self.piece: Piece = board.piece_at(self.start_position)
            self.capture: Piece = board.piece_at(self.end_position)
        elif isinstance(board, list):
            start_piece_string = board[self.start_position[0]][self.start_position[1]]
            end_piece_string = board[self.end_position[0]][self.end_position[1]]
            self.piece: Piece = Piece.from_str(start_piece_string, self.start_position)
            self.capture: Piece = Piece.from_str(end_piece_string, self.end_position)

        self.move_id: int = self.start_position[0] * 1000 + \
                            self.start_position[1] * 100 + \
                            self.end_position[0] * 10 + \
                            self.end_position[1]

    @classmethod
    def from_clicks(cls,
                    start_square: tuple[int, int],
                    end_square: tuple[int, int],
                    board) -> "Move":
        start = np.array(start_square, np.int8)
        end = np.array(end_square, np.int8)
        direction = np.subtract(end, start)

        # noinspection PyTypeChecker
        return Move(start, end, direction, board)

    def get_chess_notation(self) -> str:
        # this allows us to get a notation indicating what the move is actually doing, but isn't necessarily proper
        # "chess notation"
        return (get_rank_file(self.start_position) +
                " -> "
                + get_rank_file(self.end_position))

    def start_row(self) -> int:
        return self.start_position[0]

    def start_col(self) -> int:
        return self.start_position[1]

    def end_row(self) -> int:
        return self.end_position[0]

    def end_col(self) -> int:
        return self.end_position[1]

    def set_passant_square(self, square: np.ndarray):
        self.en_passant = square

    def set_passant_capture(self, capture: np.ndarray, board):
        self.capture = board.piece_at(capture)
        # this is probably trying to fit too much into the en_passant variable
        self.en_passant = capture

    def exposes_en_passant(self) -> bool:
        return self.en_passant is not None

    def is_en_passant_capture(self) -> bool:
        from Chess.pieces import Pawn
        return self.en_passant is not None and isinstance(self.capture, Pawn)

    def __eq__(self, other):
        return type(other) is type(self) and self.move_id == other.move_id

    def __str__(self):
        return f'MoveId: {self.move_id} | Piece: {self.piece.color}{self.piece.name} | Direction: {self.direction}'
