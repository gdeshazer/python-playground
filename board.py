import os
from typing import cast, Union

import numpy as np

from Chess.pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn, Empty


class Board:

    def __init__(self) -> None:
        self.board: list[list[Piece]] = [
            [Rook('b'), Knight('b'), Bishop('b'), Queen('b'), King('b'), Bishop('b'), Knight('b'), Rook('b')],
            [Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b')],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w')],
            [Rook('w'), Knight('w'), Bishop('w'), Queen('w'), King('w'), Bishop('w'), Knight('w'), Rook('w')],
        ]
        self.piece_index: dict[str, Piece] = {}
        self.white_to_move: bool = True
        self.move_log: list[Move] = []
        self.valid_moves: list[Move] = []
        self.white_king: King = cast(King, self.board[7][4])
        self.black_king: King = cast(King, self.board[0][4])

        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                piece = self.piece_at([r, c])
                if not isinstance(piece, Empty):
                    index = piece.id([r, c])
                    self.piece_index[index] = piece

        self.get_all_valid_moves()

    def make_move(self, move: 'Move') -> bool:
        """
        attempt to move a piece on the board if the move is valid.
        :param move: the move to make (contains start and end coordinates)
        :return: true if the move was completed, false otherwise
        """
        move_made = False

        for v_move in self.valid_moves:
            if v_move != move:
                continue

            print(f"Executing move: {v_move}")
            self.board[v_move.start_row()][v_move.start_col()] = Empty()
            self.board[v_move.end_row()][v_move.end_col()] = v_move.capture
            self.white_to_move = not self.white_to_move
            self.move_log.append(v_move)
            self.update_index(v_move)

            move_made = True

            self.get_all_valid_moves()

            break

        print(self.__str__())
        return move_made

    def undo_move(self) -> None:
        if len(self.move_log) == 0:
            return

        move = self.move_log.pop()
        self.board[move.start_row()][move.start_col()] = move.piece
        self.board[move.end_row()][move.end_col()] = move.capture

        self.update_index_with_undo(move)

        self.white_to_move = not self.white_to_move
        print(f"Undid move: {move}")

    def get_all_valid_moves(self):
        color_to_move = 'w' if self.white_to_move else 'b'
        valid_moves: list[Move] = []

        # instead of iterating through the entire board, lets only generate moves for the pieces in our index
        matching_ids = filter(lambda key: key[0] == color_to_move, self.piece_index.keys())
        for piece_id in matching_ids:
            piece = self.piece_index[piece_id]
            valid_moves.append(piece.valid_moves(self.board))

    def piece_at(self, position: Union[np.ndarray, list[int]]) -> Piece:
        return self.board[position[0]][position[1]]

    def update_index(self, move: "Move") -> None:
        start_id = move.piece.id(move.start_position)
        new_id = move.piece.id(move.end_position)
        self.piece_index[new_id] = move.piece
        self.piece_index.pop(start_id)

        if not isinstance(move.capture, Empty):
            capture_id = move.capture.id(move.end_position)
            self.piece_index.pop(capture_id)

        # debug logging
        print(f"updated index: {self.piece_index}")

    def update_index_with_undo(self, move):
        original_id = move.piece.id(move.start_position)
        undone_id = move.piece.id(move.end_position)

        self.piece_index.pop(undone_id, None)
        self.piece_index[original_id] = move.piece

        if not isinstance(move.capture, Empty):
            capture_id = move.capture.id(move.end_position)
            self.piece_index[capture_id] = move.capture

        # debug logging
        print(f"updated index for undo: {self.piece_index}")

    def __str__(self):
        output = ''
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                if len(output) == 0:
                    output = f"[ {self.piece_at([r, c])}"
                elif c == len(self.board[r]) - 1:
                    output = f"{output},{self.piece_at([r, c])}"
                    if r == len(self.board) - 1:
                        output = output + " ]"
                    else:
                        output = output + os.linesep
                else:
                    output = f"{output},{self.piece_at([r, c])}"

        return output


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
                 start_square: np.ndarray[np.int8],
                 end_square: np.ndarray[np.int8],
                 direction: np.ndarray[np.int8],
                 board) -> None:
        self.start_position: np.ndarray[np.int8] = start_square
        self.end_position: np.ndarray[np.int8] = end_square
        self.direction: np.ndarray[np.int8] = direction

        # while it would be better to import the board type here, we get a circular import warning which python
        # can't figure out.  once the new board's logic is validated to be functional, we might be able to switch back
        # to using isinstance(board, Board)
        if callable(getattr(board, 'piece_at', None)):
            self.piece: Piece = board.piece_at(self.start_position)
            self.capture: Piece = board.piece_at(self.end_position)
        elif isinstance(board, list):
            start_piece_string = board[self.start_position[0]][self.start_position[1]]
            end_piece_string = board[self.end_position[0]][self.end_position[1]]
            self.piece: Piece = Piece.from_str(start_piece_string)
            self.capture: Piece = Piece.from_str(end_piece_string)

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
        return (self.get_rank_file(self.start_position) +
                " -> "
                + self.get_rank_file(self.end_position))

    def get_rank_file(self, position: np.ndarray[np.int8]) -> str:
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

    def __str__(self):
        return f'MoveId: {self.move_id} | Piece: {self.piece.color}{self.piece.name} | Direction: {self.direction}'
