import threading
import time
import unittest

from Chess.board import Board
from Chess.chessEngine import GameState


def run_operation(times, operation):
    start = time.perf_counter(), time.process_time()

    for _ in range(times):
        operation()

    end = time.perf_counter(), time.process_time()
    return end[0] - start[0], end[1] - start[1]


def print_result(attempts, func_name, run_time):
    print(f'{attempts} attempts -> {func_name}: real {run_time[0]}\tcpu {run_time[1]}')


class PerformanceComparisonTests(unittest.TestCase):

    def setUp(self):
        self.new_board = Board.new_from_fen("2k3r1/8/8/8/r7/3K4/8/1R5R w - - 0 1")
        self.old_board = GameState()
        self.old_board.board = [
            ["--", "--", "bK", "--", "--", "--", "bR", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["bR", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "wK", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "wR", "--", "--", "--", "--", "--", "wR"],
        ]
        self.old_board.whiteToMove = True

    def test_king_move_generation(self):
        def old_operation():
            for attempts in [10, 10, 100, 1000, 10000]:
                old_generation = run_operation(attempts, self.run_old_king_generation)
                print_result(attempts, self.run_old_king_generation.__name__, old_generation)

        def new_operation():
            for attempts in [10, 10, 100, 1000, 10000]:
                new_generation = run_operation(attempts, self.run_new_king_generation)
                print_result(attempts, self.run_new_king_generation.__name__, new_generation)

        old_thread = threading.Thread(target=old_operation)
        new_thread = threading.Thread(target=new_operation)

        old_thread.start()
        new_thread.start()

        old_thread.join()
        new_thread.join()

    def test_all_generation(self):
        def old_operation():
            for attempts in [10, 10, 100, 1000, 10000]:
                old_generation = run_operation(attempts, self.run_old_generation)
                print_result(attempts, self.run_old_generation.__name__, old_generation)

        def new_operation():
            for attempts in [10, 10, 100, 1000, 10000]:
                new_generation = run_operation(attempts, self.run_new_generation)
                print_result(attempts, self.run_new_generation.__name__, new_generation)

        old_thread = threading.Thread(target=old_operation)
        new_thread = threading.Thread(target=new_operation)

        old_thread.start()
        new_thread.start()

        old_thread.join()
        new_thread.join()

    def run_new_king_generation(self):
        king = self.new_board.current_king()
        king.valid_moves(self.new_board, king.position)

    def run_old_king_generation(self):
        self.old_board.get_king_moves(5, 3, [])

    def run_new_generation(self):
        self.new_board.get_all_valid_moves()

    def run_old_generation(self):
        self.old_board.get_valid_moves()


if __name__ == '__main__':
    unittest.main()
