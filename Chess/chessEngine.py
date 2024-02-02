"""
this class is responsible for storing all the information about the current state of the game.  also responsible for
determining valid moves for the state and tracking a log of moves.

The initial implementation for the board lives here, and is in the process of being migrated to the board.py file
instead in an effort to try and make the code a bit easier to follow.
"""
from typing import List, Tuple, Union

import numpy

from Chess.move import Move
from Chess.pin import Pin

type MovePair = tuple[numpy.ndarray[numpy.int8], numpy.ndarray[numpy.int8]]


class GameState:
    """
    This stores the current state of the board
    """

    def __init__(self) -> None:
        # using numpy arrays might be faster here

        # board is 8x8 2d list where each element has two characters
        # the characters are coded so that the first character is the color of the piece (w -> white and b-> black)
        # and the second character is the piece type:
        #   r -> rook, n -> knight, q -> queen, k -> king, b -> bishop, p -> pawn
        # and an empty space is denoted by --
        # board is drawn from white's perspective
        self.board: List[List[str]] = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]

        self.whiteToMove: bool = True
        self.moveLog: List[Move] = []
        self.move_funcs = {'P': self.get_pawn_moves,
                           'R': self.get_rook_moves,
                           'N': self.get_knight_moves,
                           'B': self.get_bishop_moves,
                           'Q': self.get_queen_moves,
                           'K': self.get_king_moves}
        self.whiteKingLocation: numpy.ndarray = numpy.array([7, 4], numpy.int8)
        self.blackKingLocation: numpy.ndarray = numpy.array([0, 4], numpy.int8)
        self.in_check: bool = False
        self.pins: List[Pin] = []
        self.checks: List[MovePair] = []

    def make_move(self, move: Move) -> None:
        """
        takes a move as a parameter and does the move, does not work for castling, pawn promotion, and en-passant
        :param move:
        :return: none
        """
        self.board[move.start_row()][move.start_col()] = "--"
        self.board[move.end_row()][move.end_col()] = move.piece.full_name()
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove

        if move.piece.full_name() == 'wK':
            self.whiteKingLocation = move.end_position
        elif move.piece.full_name() == 'bK':
            self.blackKingLocation = move.end_position

    def undo_move(self) -> None:
        """
        undoes the last move
        :return: void
        """
        if len(self.moveLog) != 0:
            move = self.moveLog.pop()
            self.board[move.start_row()][move.start_col()] = move.piece.full_name()
            self.board[move.end_row()][move.end_col()] = move.capture.full_name()
            self.whiteToMove = not self.whiteToMove

            if move.piece.full_name() == 'wK':
                self.whiteKingLocation = move.start_position
            elif move.piece.full_name() == 'bK':
                self.blackKingLocation = move.start_position

    def get_valid_moves(self) -> List[Move]:
        """
        get all valid moves with checks enabled (for example this filters out moves where a piece would be creating
        a check or checkmate condition by moving)
        :return: List[Move]
        """
        moves: List[Move] = []
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()

        king_position = self.whiteKingLocation if self.whiteToMove else self.blackKingLocation

        if self.in_check:
            if len(self.checks) == 1:  # only one check, valid moves are to block it or move the king
                possible_moves = self.get_all_possible_moves()
                check = self.checks[0]
                check_move = check[0]
                check_direction = check[1]
                checking_piece = self.board[check_move[0]][check_move[1]]
                valid_squares = []

                # if the piece is a knight the only valid moves are moving the king or taking the knight
                if checking_piece[1] == 'N':
                    valid_squares = [check_move]
                else:
                    for i in range(1, 8):
                        mag_vector = numpy.multiply(check_direction, i)
                        valid_square = numpy.add(king_position, mag_vector)
                        valid_squares.append(valid_square)

                        if numpy.array_equal(valid_square, check_move):
                            break

                for move in possible_moves:
                    move_in_valid_list = self.is_in(move.end_position, valid_squares)
                    if move.piece.name == 'K':
                        if not move_in_valid_list:
                            moves.append(move)
                        else:
                            continue

                    elif move_in_valid_list:
                        moves.append(move)

            # double check means the king has to move
            else:
                self.get_king_moves(king_position[0], king_position[1], moves)
        else:
            moves = self.get_all_possible_moves()

        if len(self.pins) != 0:
            for pin in self.pins:
                moves = filter(lambda possible_pin_move: not pin.move_is_pinned(possible_pin_move), moves)

        return moves

    def is_in(self, target: numpy.ndarray, list_of_elements: list[numpy.ndarray]) -> bool:
        for element in list_of_elements:
            if numpy.array_equal(target, element):
                return True
        return False

    def check_for_pins_and_checks(self) -> Tuple[bool, List[Pin], List[MovePair]]:
        """
        check for potential pins, and checks given the current position of the player's king
        :return: Tuple[bool, List[Pin], List[MovePair]]
        """
        pins: List[Pin] = []
        checks: List[MovePair] = []
        in_check: bool = False

        if self.whiteToMove:
            enemy_color = 'b'
            ally_color = 'w'
            start_position: numpy.ndarray = self.whiteKingLocation

        else:
            enemy_color = 'w'
            ally_color = 'b'
            start_position: numpy.ndarray = self.blackKingLocation

        # this works by looking at the various directions a piece could approach the king and checking if that direction
        # is blocked by an allied piece and if there is a piece in the same direction which can move towards the king.
        # if both statements are true (there's a friendly piece around the king, and an enemy piece which could attack
        # in the same direction as the aly piece, then the aly piece is considered "pinned"

        directions = numpy.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]], numpy.int8)
        for direction in directions:
            possible_pin: Union[Pin, None] = None
            for magnitude in range(1, 8):
                dir_vector: numpy.ndarray[numpy.int8] = numpy.multiply(direction, magnitude)
                move: numpy.ndarray[numpy.int8] = numpy.add(start_position, dir_vector)

                # we are outside the board
                if numpy.any(move > 7) or numpy.any(move < 0):
                    break

                target = self.board[move[0]][move[1]]
                # since we have some points where we "move" the king without actually moving him on the board, we need
                # to make sure we don't accidentally mark the king as a pin
                if target[0] == ally_color and target[1] != 'K':
                    if possible_pin is None:
                        possible_pin = Pin(target, move, direction)
                    else:
                        break
                elif target[0] == enemy_color:
                    piece_type = target[1]

                    # there are 5 cases to check
                    # 1. an orthogonal rook (up, down, left, or right)
                    # 2. a diagonal bishop (left-up diag, right-up diag, left-down diag, right-down diag)
                    # 3. a pawn which is 1 diagonal away
                    # 4. any direction contains a queen
                    # 5. any direction 1 square away is enemy king (prevents moving king next to other king)

                    # unit vectors which are in cardinal direction will always have a product of 0
                    direction_product = numpy.prod(direction)
                    orthogonal_rook = direction_product == 0 and piece_type == 'R'
                    diagonal_bishop = abs(direction_product) == 1 and piece_type == 'B'
                    diagonal_pawn = magnitude == 1 and (
                            (enemy_color == 'w' and direction[0] == -1)
                            or (enemy_color == 'b' and direction[0] == 1)
                    )
                    queen_present = piece_type == 'Q'
                    near_king = magnitude == 1 and piece_type == 'K'

                    if orthogonal_rook or diagonal_bishop or diagonal_pawn or queen_present or near_king:
                        # no piece is blocking so we are probably in check
                        if possible_pin is None:
                            in_check = True
                            checks.append((move, direction))
                            break

                        # pin is in the way
                        else:
                            pins.append(possible_pin)
                    else:
                        break

        # knight checks
        knight_moves = numpy.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], numpy.int8)
        for knight_direction in knight_moves:
            move: numpy.ndarray[numpy.int8] = numpy.add(start_position, knight_direction)

            # we are outside the board
            if numpy.any(move > 7) or numpy.any(move < 0):
                continue

            target_piece = self.board[move[0]][move[1]]

            if target_piece[0] == enemy_color and target_piece[1] == 'N':
                in_check = True
                checks.append((move, knight_direction))

        return in_check, pins, checks

    def get_all_possible_moves(self) -> List[Move]:
        """
        get all possible moves without checks
        :return: list of all possible moves
        """
        # i wonder if it would be better to set the pieces up as individual objects that know their location and their
        # own valid moves
        moves = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                turn = self.board[row][col][0]

                if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[row][col][1]
                    self.move_funcs[piece](row, col, moves)

        return moves

    def get_pawn_moves(self, row: int, col: int, moves: list[Move]) -> None:
        """
        get all possible pawn moves from a given starting row and col, then append those moves to the provided list of
        moves
        :param row: the starting row
        :param col: the starting column
        :param moves: List of all currently calculated moves
        :return: None
        """
        # noinspection PyTypeChecker
        start: numpy.ndarray[numpy.int8] = numpy.array([row, col], numpy.int8)

        # noinspection PyTypeChecker
        white_directions: numpy.ndarray[numpy.int8] = numpy.array([[-1, 0], [-1, 1], [-1, -1]], numpy.int8)

        # noinspection PyTypeChecker
        black_directions: numpy.ndarray[numpy.int8] = numpy.array([[1, 0], [1, 1], [1, -1]], numpy.int8)

        # white pawns start on 6, black pawns start on 1
        if self.whiteToMove:
            self.calculate_pawn_moves(moves, start, white_directions, row == 6, 'b')

        elif not self.whiteToMove:
            self.calculate_pawn_moves(moves, start, black_directions, row == 1, 'w')

    def calculate_pawn_moves(self,
                             moves: List[Move],
                             start: numpy.ndarray[numpy.int8],
                             directions: numpy.ndarray[numpy.int8],
                             is_in_start: bool,
                             capture_color: str) -> None:
        """
        Given a starting point, a set of directions, the starting row, and capture color, calculate allowable pawn moves
        :param moves: the current list of allowed moves to add to
        :param start: the starting point for the pawn
        :param directions: a list of directions to move in
        :param is_in_start: indicates in the pawn is currently in a starting position on the board or not
        :param capture_color: which color the current pawn is allowed to capture
        :return: none
        """
        for direction in directions:
            direction_prod = numpy.prod(direction)
            move = numpy.add(start, direction)

            # we are outside the board
            if numpy.any(move > 7) or numpy.any(move < 0):
                break

            target = self.board[move[0]][move[1]]

            # we are moving vertically
            if direction_prod == 0:
                if target == "--":
                    moves.append(Move(start, move, direction, self.board))

                # pawns in their starting row can move two spaces forward
                if is_in_start:
                    additional_move = numpy.add(move, direction)
                    new_target = self.board[additional_move[0]][additional_move[1]]
                    if new_target == "--":
                        moves.append(Move(start, additional_move, direction, self.board))

            # moving in diagonals to capture
            elif target[0] == capture_color:
                moves.append(Move(start, move, direction, self.board))

    def get_rook_moves(self, row: int, col: int, moves: List[Move]) -> None:
        directions = numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]], numpy.int8)
        magnitudes = numpy.arange(1, 8)
        self.append_moves_from_directions(row, col, directions, magnitudes, moves)

    def get_knight_moves(self, row: int, col: int, moves: List[Move]) -> None:
        # knight movements are an L shape, we could represent this as a series of vectors
        possible_moves = numpy.array([
            [2, 1],
            [2, -1],
            [-2, 1],
            [-2, -1],
            [1, 2],
            [1, -2],
            [-1, 2],
            [-1, -2]
        ], numpy.int8)

        # noinspection PyTypeChecker
        start: numpy.ndarray[numpy.int8] = numpy.array([row, col], numpy.int8)

        for direction in possible_moves:
            move = numpy.add(start, direction)

            # move is invalid
            if numpy.any(move > 7) or numpy.any(move < 0):
                continue

            if self.whiteToMove and self.board[move[0]][move[1]][0] != 'w':
                moves.append(Move(start, move, direction, self.board))

            if not self.whiteToMove and self.board[move[0]][move[1]][0] != 'b':
                moves.append(Move(start, move, direction, self.board))

    def get_bishop_moves(self, row: int, col: int, moves: List[Move]) -> None:
        diagonal_directions = numpy.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], numpy.int8)
        magnitudes = numpy.arange(1, 8)
        self.append_moves_from_directions(row, col, diagonal_directions, magnitudes, moves)

    def get_king_moves(self, row: int, col: int, moves: List[Move]) -> None:
        directions = numpy.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]], numpy.int8)

        # noinspection PyTypeChecker
        start: numpy.ndarray[numpy.int8] = numpy.array([row, col], numpy.int8)

        for direction in directions:
            move = numpy.add(start, direction)

            if numpy.any(move > 7) or numpy.any(move < 0):
                continue

            # we need to check if the move places us in check and not add the move if it puts us in a bad position
            # pretend we moved the king by setting the king location and then run the check_for_pins_and_checks method
            if self.whiteToMove:
                self.whiteKingLocation = move
            else:
                self.blackKingLocation = move

            in_check, pins, checks = self.check_for_pins_and_checks()
            if in_check:
                # reset and skip the direction
                if self.whiteToMove:
                    self.whiteKingLocation = numpy.array((row, col), numpy.int8)
                else:
                    self.blackKingLocation = numpy.array((row, col), numpy.int8)

                continue

            target = self.board[move[0]][move[1]][0]
            if self.whiteToMove and target != 'w':
                moves.append(Move(start, move, direction, self.board))

            elif not self.whiteToMove and target != 'b':
                moves.append(Move(start, move, direction, self.board))

    def get_queen_moves(self, row: int, col: int, moves: List[Move]) -> None:
        self.get_bishop_moves(row, col, moves)
        self.get_rook_moves(row, col, moves)

    def append_moves_from_directions(self,
                                     row: int,
                                     col: int,
                                     diagonal_directions: numpy.ndarray,
                                     magnitudes: numpy.ndarray,
                                     moves: List[Move]) -> None:
        enemy_color = 'b' if self.whiteToMove else 'w'

        # noinspection PyTypeChecker
        start: numpy.ndarray[numpy.int8] = numpy.array([row, col], numpy.int8)

        for direction in diagonal_directions:
            for magnitude in magnitudes:
                dir_vector: numpy.ndarray[numpy.int8] = numpy.multiply(direction, magnitude)
                move_vector: numpy.ndarray[numpy.int8] = numpy.add(start, dir_vector)

                # we are outside the board
                if numpy.any(move_vector > 7) or numpy.any(move_vector < 0):
                    break

                target = self.board[move_vector[0]][move_vector[1]][0]

                if target == '-':
                    moves.append(Move(start, move_vector, direction, self.board))
                    continue
                elif target == enemy_color:
                    moves.append(Move(start, move_vector, direction, self.board))
                    break
                else:
                    break
