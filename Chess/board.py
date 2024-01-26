import os
from typing import cast, Union

import numpy as np

from Chess.move import Move
from Chess.pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn, Empty


class Board:

    def __init__(self) -> None:
        # whenever a move is made, the move should ultimately be propagated to this two-dimensional list.  this
        # ultimately is what keeps track of what piece is where and which squares are currently occupied
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
                    index = piece.position_id([r, c])
                    self.piece_index[index] = piece

        self.get_all_valid_moves()

    def make_move(self, move: Move) -> bool:
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
            self.board[v_move.end_row()][v_move.end_col()] = v_move.piece
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
        # we need to clear out the previous set of moves since we are switching turns and will have a different set of
        # moves to manage
        self.valid_moves = []
        color_to_move = 'w' if self.white_to_move else 'b'

        # instead of iterating through the entire board, lets only generate moves for the pieces in our index
        matching_ids = filter(lambda key: key[0] == color_to_move, self.piece_index.keys())
        for piece_id in matching_ids:
            piece = self.piece_index[piece_id]
            self.valid_moves += piece.valid_moves(self, piece.id_to_position(piece_id))

    def piece_at(self, position: Union[np.ndarray, list[int]]) -> Piece:
        return self.board[position[0]][position[1]]

    def update_index(self, move: "Move") -> None:
        start_id = move.piece.position_id(move.start_position)
        new_id = move.piece.position_id(move.end_position)
        self.piece_index[new_id] = move.piece
        self.piece_index.pop(start_id)

        if not isinstance(move.capture, Empty):
            capture_id = move.capture.position_id(move.end_position)
            self.piece_index.pop(capture_id)

        # debug logging
        print(f"updated index: {self.piece_index}")

    def update_index_with_undo(self, move):
        original_id = move.piece.position_id(move.start_position)
        undone_id = move.piece.position_id(move.end_position)

        self.piece_index.pop(undone_id, None)
        self.piece_index[original_id] = move.piece

        if not isinstance(move.capture, Empty):
            capture_id = move.capture.position_id(move.end_position)
            self.piece_index[capture_id] = move.capture

        # debug logging
        print(f"updated index for undo: {self.piece_index}")

    def current_fen(self):
        """
        Convert the current board state into a "FEN" string.  The FEN codifies which pieces are where for the entire
        board and is a standard notation we can use to compare with other engines or online sources.

        see: https://www.chess.com/terms/fen-chess for additional information
        :return: The FEN string
        """
        fen = ""
        for row in self.board:
            empty_count = 0
            for piece in row:
                if isinstance(piece, Empty):
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += piece.get_chess_notation()
            if empty_count > 0:
                fen += str(empty_count)
            fen += "/"
        fen = fen[:-1]  # remove the trailing "/"
        fen += " w " if self.white_to_move else " b "
        fen += "- - 0 1"  # placeholder for castling rights, en passant square, and halfmove and fullmove counters
        return fen

    def __str__(self):
        """
        convert the current board state into a semi-human readable string.  to compare different states together, the
        FEN string is a better option
        :return: a string representation of the board
        """
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
