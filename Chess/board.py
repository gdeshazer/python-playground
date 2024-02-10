import os
from typing import Union

import numpy as np

from Chess.move import Move, get_rank_file
from Chess.pieces import Piece, Rook, Knight, Bishop, Queen, King, Pawn, Empty


class Board:

    def __init__(self) -> None:
        self.white_king: King = King('w', [7, 4])
        self.black_king: King = King('b', [0, 4])

        # whenever a move is made, the move should ultimately be propagated to this two-dimensional list.  this
        # ultimately is what keeps track of what piece is where and which squares are currently occupied
        self.board: list[list[Piece]] = [
            [Rook('b'), Knight('b'), Bishop('b'), Queen('b'), self.black_king, Bishop('b'), Knight('b'), Rook('b')],
            [Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b'), Pawn('b')],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty()],
            [Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w'), Pawn('w')],
            [Rook('w'), Knight('w'), Bishop('w'), Queen('w'), self.white_king, Bishop('w'), Knight('w'), Rook('w')],
        ]
        self.piece_index: dict[str, Piece] = {}
        self.white_to_move: bool = True
        self.move_log: list[Move] = []

        # todo: maybe use a dictionary for this?
        self.valid_moves: list[Move] = []

        self.passant_square = []

        self.rebuild_piece_index()
        self.get_all_valid_moves()

    @classmethod
    def new_from_fen(cls, fen: str) -> "Board":
        """
        initialize the board from the provided FEN.  this should allow for starting the board in a specific state (good
        for testing), but that state may or may not actually be reachable
        :return: a new board based on the current fen string
        """
        # a fen should look like this (standard starting position): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        # splitting will give us [ rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR, w, KQkq, -, 0, 1 ]
        fen_parts = fen.split()

        # this will have [ rnbqkbnr, pppppppp, 8, 8, 8, 8, PPPPPPPP, RNBQKBNR ]
        row_strs = fen_parts[0].split('/')
        board = cls()
        board.board = [[Empty() for _ in range(8)] for _ in range(8)]

        for row in range(len(row_strs)):
            row_str = row_strs[row]
            insert_column = 0

            for c in range(len(row_str)):
                if row_str[c].isdigit():
                    insert_column += int(row_str[c])
                else:
                    color = 'w' if row_str[c].isupper() else 'b'
                    piece = Piece.from_str(f'{color}{row_str[c]}', [row, insert_column])
                    board.board[row][insert_column] = piece

                    if isinstance(piece, King):
                        if color == 'w':
                            board.white_king = piece
                        else:
                            board.black_king = piece

                    insert_column += 1

            if insert_column != 8:
                raise ValueError(f"Invalid FEN string: {fen}")

        board.white_to_move = fen_parts[1] == 'w'
        board.rebuild_piece_index()
        board.get_all_valid_moves()

        return board

    def rebuild_piece_index(self):
        self.piece_index = {}
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                piece = self.piece_at([r, c])
                if not isinstance(piece, Empty):
                    index = piece.position_id([r, c])
                    self.piece_index[index] = piece

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

            if v_move.exposes_en_passant():
                self.passant_square = v_move.en_passant
            else:
                self.passant_square = []

            if v_move.is_en_passant_capture():
                self.board[v_move.en_passant[0]][v_move.en_passant[1]] = Empty()
                self.passant_square = []

            # since we need the king location to figure out pins/checks, we need to just make sure we update the kings
            # position.  it might be worth doing this for all pieces instead of having the board variable for it, but for
            # now this seems like an easy option
            if isinstance(v_move.piece, King):
                v_move.piece.position = v_move.end_position

            self.get_all_valid_moves()

            move_made = True

            break

        print(f'successfully made move: {move_made}\n{self.__str__()}')
        return move_made

    def undo_move(self) -> None:
        if len(self.move_log) == 0:
            return

        move = self.move_log.pop()

        if move.is_en_passant_capture():
            self.board[move.start_row()][move.start_col()] = move.piece
            self.board[move.en_passant[0]][move.en_passant[1]] = move.capture
            self.board[move.end_position[0]][move.end_position[1]] = Empty()
        else:
            self.board[move.start_row()][move.start_col()] = move.piece
            self.board[move.end_row()][move.end_col()] = move.capture

        self.update_index_with_undo(move)

        self.white_to_move = not self.white_to_move
        self.get_all_valid_moves()
        print(f"Undid move: {move}\twhite_to_move: {self.white_to_move}")

    def get_all_valid_moves(self):
        # it's tempting to do the pin/check validation in the pieces themselves, but in order to make that work you
        # would need each piece to check one move set ahead which could be expensive to do on every move made in
        # addition to eventually having the engine pick moves for black to make
        self.find_pins_and_checks()

        # we need to clear out the previous set of moves since we are switching turns and will have a different set of
        # moves to manage
        self.valid_moves = []
        king = self.current_king()

        if king.in_check:
            if len(king.check_directions) > 1:
                self.valid_moves = king.valid_moves(self, king.position)
                return

            valid_squares = king.valid_positions_if_checked()

            possible_moves = self.get_all_moves()
            valid_moves = []
            for move in possible_moves:
                move_is_valid = self.is_in(move.end_position, valid_squares)

                # if the piece is a king, we should also allow the king to move out of check if possible. Since the king
                # piece in theory has its own rules for generating moves, we should hopefully be able to just add the
                # moves without double-checking
                #
                # Probably, this whole bit of logic should be moved into the pieces themselves, instead of having move
                # validation logic in more than one place
                if isinstance(move.piece, King) or move_is_valid:
                    valid_moves.append(move)
                else:
                    continue

            self.valid_moves = valid_moves
        else:
            self.valid_moves = self.get_all_moves()

        king.reset_check()
        for piece_id in filter(lambda key: key[0] == king.color, self.piece_index.keys()):
            self.piece_index[piece_id].reset_pin()

    def is_in(self, target: np.ndarray, list_of_elements: list[np.ndarray]) -> bool:
        for element in list_of_elements:
            if np.array_equal(target, element):
                return True
        return False

    def get_all_moves(self) -> list[Move]:
        # instead of iterating through the entire board, lets only generate moves for the pieces in our index
        color_to_move = 'w' if self.white_to_move else 'b'
        matching_ids = filter(lambda key: key[0] == color_to_move, self.piece_index.keys())
        valid_moves = []

        for piece_id in matching_ids:
            piece = self.piece_index[piece_id]
            valid_moves += piece.valid_moves(self, piece.id_to_position(piece_id))

        return valid_moves

    def find_pins_and_checks(self) -> list:
        directions = Piece.ALL_DIRECTIONS
        magnitudes = np.arange(1, 8)
        king = self.current_king()
        start = king.position

        pins = []

        # moving in all directions away from the king, check for:
        # - friendly piece (but if there are two in the same direction skip to next direction)
        # - enemy piece
        #     - if enemy can attack and we found a friendly piece in the way, that friendly piece is pinned
        #     - if the enemy can attack and there is not a friendly piece, we are in check
        for direction in directions:
            possible_pin = None
            for magnitude in magnitudes:
                # todo: maybe move this into a function?
                end = np.add(start, np.multiply(direction, magnitude))

                # we are outside the board
                if np.any(end > 7) or np.any(end < 0):
                    break

                target = self.piece_at(end)

                if isinstance(target, Empty):
                    continue

                if target.color == king.color:
                    if possible_pin is None:
                        possible_pin = (target, direction)
                    else:
                        # go to next magnitude in this direction to check for attacking piece
                        continue
                else:
                    can_attack = target.can_attack_in_direction(direction) and magnitude in target.magnitudes()

                    if can_attack:
                        if possible_pin is None:
                            # todo might need to flip the direction of the vector here
                            king.check_directions.append(direction)
                            king.in_check = True
                            break
                        else:
                            pins.append(possible_pin)
                    else:
                        break

        # we have to check knights separately since they can hop pieces
        for knight_direction in Knight.DIRECTIONS:
            end = np.add(start, knight_direction)

            # we are outside the board
            if np.any(end > 7) or np.any(end < 0):
                break

            target = self.piece_at(end)

            if isinstance(target, Knight) and target.color != king.color:
                king.in_check = True
                king.check_directions.append(knight_direction)

        # todo: not sure if better to tell a piece it is pinned, or if better to return pin list
        for pin in pins:
            pin[0].is_pinned = True
            pin[0].pin_direction = pin[1]

        return pins

    def current_king(self):
        return self.white_king if self.white_to_move else self.black_king

    def piece_at(self, position: Union[np.ndarray, list[int]]) -> Piece:
        return self.board[position[0]][position[1]]

    def update_index(self, move: "Move") -> None:
        start_id = move.piece.position_id(move.start_position)
        new_id = move.piece.position_id(move.end_position)
        self.piece_index[new_id] = move.piece
        self.piece_index.pop(start_id)

        if not isinstance(move.capture, Empty):
            if move.is_en_passant_capture():
                capture_id = move.capture.position_id(move.en_passant)
            else:
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
            if move.is_en_passant_capture():
                capture_id = move.capture.position_id(move.en_passant)
            else:
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
        fen += "-"  # place holder for castling

        if len(self.passant_square) > 0:
            fen += f" {get_rank_file(self.passant_square)} "
        else:
            fen += " - "

        fen += "0 1"  # halfmove and fullmove counters
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
