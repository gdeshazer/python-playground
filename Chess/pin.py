import numpy

from Chess.move import Move


class Pin:
    def __init__(self, piece: str, position: numpy.ndarray, pin_direction: numpy.ndarray) -> None:
        self.piece = piece
        self.position = position
        self.pin_direction = pin_direction

    def move_is_pinned(self, move: Move) -> bool:
        """
        This method allows us to test if the current move matches a pinned piece.  If it matches, we return true and the
        move should be disallowed.

        A match occurs when the move defines the same piece, the same starting position as the pin, and the direction
        is in any direction except the direction of the pin itself.  This is because, moving a piece in the same direction
        as a pin, is acceptable (for example you could move a "pinned" bishop along its diagonal path  as long as the
        path matches the pinned direction, any other moves would be exposing your king to check/checkmate)
        :param move: the move to compare against
        :return: true if the move should not be allowed
        """
        matching_piece = move.piece.full_name() == self.piece

        if not matching_piece:
            return False

        matching_start = numpy.array_equal(move.start_position, self.position)

        if not matching_start:
            return False

        # the direction matches if the direction is exactly the same as the pin direction, or if the direction is in the
        # opposite direction of the pin direction
        matching_direction = (numpy.array_equal(move.direction, self.pin_direction) or
                              numpy.array_equal(move.direction, numpy.multiply(self.pin_direction, -1)))

        # since we returned if either of the above statements were false, we don't need to recheck them for truthiness
        # here
        return not matching_direction
