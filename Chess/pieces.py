import math
from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Piece(ABC):
    DIAGONALS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], np.int8)
    UP_DOWN = np.array([[1, 0], [-1, 0]])
    LEFT_RIGHT = np.array([[0, 1], [0, -1]])
    ALL_DIRECTIONS = np.concatenate((DIAGONALS, UP_DOWN, LEFT_RIGHT))

    def __init__(self, color: str, name: str) -> None:
        self._attack_unit_vectors = []
        self.color = color
        self.name = name
        self.is_pinned = False
        self.pin_direction = []
        # may need to keep an eye on the size of this object.  the long the game is played the larger this will get
        self._attacks_from_positions = {}

    @classmethod
    def from_str(cls, name: str, position) -> "Piece":
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
            return King(color, position)
        elif piece_type == 'P':
            return Pawn(color)
        elif piece_type == "-":
            return Empty()
        else:
            return Empty()

    # todo: the pins value needs to be reset after the valid moves have been generated...or the board needs to go through
    #       each piece and reset any pinned pieces
    @abstractmethod
    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        pass

    @abstractmethod
    def attack_directions(self) -> np.ndarray:
        pass

    @abstractmethod
    def magnitudes(self):
        pass

    def can_attack_in_direction(self, direction) -> bool:
        # turns out using the cross product for this operation is really fast compared to checking if the unit direction
        # vector is contained in the array of unit direction vectors the piece could move in
        for attack_direction in self.attack_directions():
            # we have two parallel arrays
            cross_product = np.cross(attack_direction, direction)
            if np.size(cross_product) > 1:
                # something terrible has happened
                print(
                    f"Cross product for direction attack has failed for some unknown reason: {attack_direction} x {direction}")
            elif cross_product.item() == 0:
                return True

        return False

    def can_attack_square(self, current_position, target, board) -> bool:
        """
        the goal of this function is to determine if the current piece at some current position can capture a piece on
        target square.

        this is determined in a couple different steps:
        1. if the piece can move in a direction parallel to the direction the target is in
            - the direction to the target is determined by subtracting the target's coordinates from the current
              coordinates (this gives us a 'vector' indicating how to get from current position to target)
            - the piece can potentially move to the target if the direction is parallel with one or more of the current
              pieces possible attack directions
        2. Find which directions (at most there will only be two) which match the direction to the target, and then
           generate coordinates from the current position along that direction until we either hit the target or we hit
           some other piece
            - if we hit the target then we know the piece can move to that target
            - if we never hit the target, or we hit a piece before the target, then we know we can't attack the target
              square
        """
        cache = self._known_attack(current_position, target)
        if cache[0]:
            return cache[1]

        target_direction = np.subtract(target, current_position)
        target_magnitude = math.trunc(np.linalg.norm(target_direction))

        if not np.any(self.magnitudes() >= target_magnitude):
            self._cache_attack(current_position, target, False)
            return False

        # this might seem strange, but in cases where we are looking ahead a move (like in the case of the king) this
        # could happen when the king could move to the current square occupied by the current piece.
        if np.array_equal(target_direction, target):
            self._cache_attack(current_position, target, False)
            return False

        unit_vector_dir = target_direction / np.linalg.norm(target_direction)

        if not self.can_attack_in_direction(target_direction):
            self._cache_attack(current_position, target, False)
            return False

        selected_directions = []
        for direction in self.attack_directions():
            unit_dir = direction / np.linalg.norm(direction)
            inverse_dir = np.multiply(unit_dir, -1)

            # in theory, we would only need to check one direction but i wasn't able to think of a good way to figure
            # out how to determine what the sign of the direction should be (ie are we moving in a more positive
            # direction or a more negative direction)
            if np.all(np.isclose(unit_vector_dir, unit_dir)) or np.all(np.isclose(unit_vector_dir, inverse_dir)):
                selected_directions.append(direction)

        if len(selected_directions) == 0:
            self._cache_attack(current_position, target, False)
            return False

        for direction in selected_directions:
            for magnitude in self.magnitudes():
                dir_vector = np.multiply(direction, magnitude)
                end_position = np.add(current_position, dir_vector)

                # check out of bounds
                if np.any(end_position > 7) or np.any(end_position < 0):
                    break

                if np.array_equal(end_position, target):
                    self._cache_attack(current_position, target, True)
                    return True

                end_target = board.piece_at(end_position)

                # if it isn't an empty piece we can know this direction is blocked, so we can go to the next direction
                if not isinstance(end_target, Empty):
                    break

    def _known_attack(self, current_position, target) -> tuple[bool, Union[bool, None]]:
        key = self._attack_key(current_position, target)

        if key in self._attacks_from_positions:
            return True, self._attacks_from_positions[key]

        return False, None

    def _cache_attack(self, current_position, target, can_attack) -> None:
        key = self._attack_key(current_position, target)
        self._attacks_from_positions[key] = can_attack

    @staticmethod
    def _attack_key(current_position, target) -> str:
        return f'{current_position[0]}{current_position[1]}{target[0]}{target[1]}'

    def position_id(self, position: Union[np.ndarray, list[int]]) -> str:
        return f'{self.color}{self.name}-{position[0]}{position[1]}'

    # todo: there should be a better way to handle this to, maybe the piece should store it's current position?
    @staticmethod
    def id_to_position(id_str: str) -> np.ndarray[np.int8]:
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
        if not self.is_pinned:
            directions_to_move = directions
        else:
            # todo: this could be a problem if we flip the vector and the piece can't move in the opposite direction
            directions_to_move = [self.pin_direction, np.multiply(self.pin_direction, -1)]

        for direction in directions_to_move:
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

    def attack_unit_vectors(self):
        if len(self._attack_unit_vectors) > 0:
            return self._attack_unit_vectors

        self._attack_unit_vectors = [direction / np.linalg.norm(direction) for direction in self.attack_directions()]
        return self._attack_unit_vectors

    def reset_pin(self):
        self.pin_direction = []
        self.is_pinned = False

    def __str__(self):
        return self.full_name()


class Empty(Piece):
    def __init__(self) -> None:
        super().__init__('-', '-')

    def valid_moves(self, board: list[list["Piece"]], position: np.ndarray) -> list:
        return []

    def attack_directions(self) -> np.ndarray:
        return np.array([])

    def magnitudes(self):
        return np.array([])


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

    def attack_directions(self) -> np.ndarray:
        if self.color == 'w':
            return np.array([[-1, 1], [-1, -1]])
        else:
            return np.array([[1, 1], [1, -1]])

    def calculate_pawn_moves(self,
                             board,
                             start: np.ndarray[np.int8],
                             directions: np.ndarray,
                             is_in_start: bool,
                             capture_color: str) -> list:
        from Chess.move import Move
        moves: list = []
        if not self.is_pinned:
            directions_to_move = directions
        else:
            # todo: this could be a problem if we flip the vector and the piece can't move in the opposite direction
            directions_to_move = [self.pin_direction, np.multiply(self.pin_direction, -1)]

        for direction in directions_to_move:
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

    def magnitudes(self):
        return np.array([1])


class Bishop(Piece):
    DIRECTIONS = Piece.DIAGONALS
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'B')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)

    def attack_directions(self) -> np.ndarray:
        return self.DIRECTIONS

    def magnitudes(self):
        return self.MAGNITUDES


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
    # this isn't really the correct magnitude value for the knight piece.
    MAGNITUDES = np.array([1])

    def __init__(self, color: str) -> None:
        super().__init__(color, 'N')
        self._actual_magnitudes = np.array([], dtype=np.int8)

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)

    def attack_directions(self) -> np.ndarray:
        return self.DIRECTIONS

    def can_attack_square(self, current_position, target, board) -> bool:
        """
        For knights, we have to take a different approach compared to the other pieces and this is because we know that
        knights can only move to certain squares.  Given a current position for a knight, we can easily figure out
        whether that square is one the knight can move to or not by just calculating the direction to the target, and if
        the direction matches one of the directions the knight can move in, then we can attack the square, otherwise,
        we cannot.
        """
        cache = self._known_attack(current_position, target)
        if cache[0]:
            return cache[1]

        target_direction = np.subtract(target, current_position)

        for direction in self.DIRECTIONS:
            if np.array_equal(target_direction, direction):
                self._cache_attack(current_position, target, True)
                return True

        # we've exhausted all options, so we must not be able to attack the square
        self._cache_attack(current_position, target, False)
        return False

    def magnitudes(self):
        if len(self._actual_magnitudes) == 0:
            for direction in self.DIRECTIONS:
                magnitude = int(math.trunc(np.linalg.norm(direction)))
                if not np.any(self._actual_magnitudes == magnitude):
                    self._actual_magnitudes = np.append(self._actual_magnitudes, magnitude)

        return self._actual_magnitudes


class Rook(Piece):
    DIRECTIONS = np.concatenate((Piece.UP_DOWN, Piece.LEFT_RIGHT))
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'R')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)

    def attack_directions(self) -> np.ndarray:
        return self.DIRECTIONS

    def magnitudes(self):
        return self.MAGNITUDES


class Queen(Piece):
    DIRECTIONS = Piece.ALL_DIRECTIONS
    MAGNITUDES = np.arange(1, 8)

    def __init__(self, color: str) -> None:
        super().__init__(color, 'Q')

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        return self.build_moves_from_directions(position, self.DIRECTIONS, self.MAGNITUDES, board)

    def attack_directions(self) -> np.ndarray:
        return self.DIRECTIONS

    def magnitudes(self):
        return self.MAGNITUDES


class King(Piece):
    DIRECTIONS = Piece.ALL_DIRECTIONS
    MAGNITUDES = np.array([1])

    def __init__(self, color: str, position) -> None:
        super().__init__(color, 'K')
        self.in_check = False
        self.check_directions = []
        self.position = position
        self._unit_vectors = []

    def valid_moves(self, board, position: np.ndarray[np.int8]) -> list:
        possible_moves = self.build_moves_from_directions(position, self.ALL_DIRECTIONS, self.MAGNITUDES, board)
        enemy_positions = list(filter(lambda key: key[0] != self.color, board.piece_index.keys()))

        # todo: would be kinda cool if we could figure out right off the bat if a given piece is too far away and just
        #       eliminate it from the pool.  could save a few cycles of extra compute time later on
        valid_moves = []
        for move in possible_moves:
            is_attacked = False
            for enemy_index in enemy_positions:
                enemy: Piece = board.piece_index[enemy_index]

                is_attacked = is_attacked or enemy.can_attack_square(enemy.id_to_position(enemy_index),
                                                                     move.end_position,
                                                                     board)

                if is_attacked:
                    break

            if not is_attacked:
                valid_moves.append(move)

        return valid_moves

    def attack_directions(self) -> np.ndarray:
        return self.DIRECTIONS

    def valid_positions_if_checked(self) -> list[np.ndarray]:
        if not self.in_check:
            return []

        valid_positions = []
        for check_direction in self.check_directions:
            for magnitude in np.arange(1, 8):
                position = np.add(self.position, np.multiply(check_direction, magnitude))

                if np.any(position > 7) or np.any(position < 0):
                    break

                valid_positions.append(position)

        return valid_positions

    def reset_check(self) -> None:
        self.in_check = False
        self.check_directions = []

    def magnitudes(self):
        return self.MAGNITUDES
