from typing import List

import numpy


class Move:
    """
    The move class is responsible for containing information about a possible move
    """

    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    fields_to_col = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    col_to_field = {v: k for k, v in fields_to_col.items()}

    # for some reason the type checker won't allow for the list to be fully coerced into the numpy types
    # noinspection PyTypeChecker
    def __init__(self,
                 start_square: numpy.ndarray[numpy.int8],
                 end_square: numpy.ndarray[numpy.int8],
                 direction: numpy.ndarray[numpy.int8],
                 board: List[List[str]]) -> None:
        self.start_position: numpy.ndarray[numpy.int8] = start_square
        self.end_position: numpy.ndarray[numpy.int8] = end_square
        self.direction: numpy.ndarray[numpy.int8] = direction

        self.piece_moved: str = board[self.start_position[0]][self.start_position[1]]

        # this could be an empty square "--"
        self.piece_captured: str = board[self.end_position[0]][self.end_position[1]]

        self.move_id: int = self.start_position[0] * 1000 + \
                            self.start_position[1] * 100 + \
                            self.end_position[0] * 10 + \
                            self.end_position[1]

        print(f'MoveId: {self.move_id} | Piece: {self.piece_moved} | Direction: {self.direction}')

    @classmethod
    def from_clicks(cls, start_square: tuple[int, int], end_square: tuple[int, int], board: list[list[str]]) -> "Move":
        start = numpy.array(start_square, numpy.int8)
        end = numpy.array(end_square, numpy.int8)
        direction = numpy.subtract(end, start)

        # noinspection PyTypeChecker
        return Move(start, end, direction, board)

    def get_chess_notation(self) -> str:
        # this allows us to get a notation indicating what the move is actually doing, but isn't necessarily proper
        # "chess notation"
        return (self.get_rank_file(self.start_position) +
                " -> "
                + self.get_rank_file(self.end_position))

    def get_rank_file(self, position: numpy.ndarray[numpy.int8]) -> str:
        return self.col_to_field[position[1]] + self.rows_to_ranks[position[0]]

    def start_row(self) -> int:
        return self.start_position[0]

    def start_col(self) -> int:
        return self.start_position[1]

    def end_row(self) -> int:
        return self.end_position[0]

    def end_col(self) -> int:
        return self.end_position[1]

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.move_id == other.move_id

        return False
