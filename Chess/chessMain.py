"""
This is the main driver file and is responsible for handling user input and displaying current game state

see https://www.youtube.com/watch?v=EnYui0e73Rs&list=PLBwF487qi8MGU81nDGaeNE1EnNEPYWKY_&ab_channel=EddieSharick for some
the guide this is derived from.  The images for the chess pieces are also sourced from the video series description.
"""

import pygame

from Chess.board import Board
from Chess.move import Move

WIDTH = HEIGHT = 512
DIMENSION = 8  # boards are 8 by 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15  # for animations later on
IMAGES = {}


def resolve_path(filename):
    """
    Resolve the path of the given filename relative to the current file
    :param filename: the name of the file
    :return: the resolved path
    """
    import os

    base_path = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(base_path, filename)
    return os.path.abspath(relative_path)


def load_images():
    """
    Initialize global dictionary of images, called exactly once in main
    """
    pieces = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces:
        path = resolve_path(f'./images/{piece}.png')
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(path), (SQ_SIZE, SQ_SIZE))


def main():
    """
    Main driver for the game.  handles user input and updating graphics
    :return: void
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color("white"))

    # game_state = GameState()
    board = Board()

    load_images()  # only do this once
    running = True

    selected_square = ()  # keep track of last user click in a tuple
    player_clicks = []  # keep track of player clicks, (two tuples, [(6,4), (4,4)] )

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            # mouse handler
            elif e.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE

                # this is when the user clicks the same square twice, just reset
                if selected_square == (row, col):
                    selected_square = ()
                    player_clicks = []
                else:
                    selected_square = (row, col)
                    player_clicks.append(selected_square)

                if len(player_clicks) == 2:
                    move = Move.from_clicks(player_clicks[0], player_clicks[1], board)
                    print(move.get_chess_notation())

                    if board.make_move(move):
                        selected_square = ()
                        player_clicks = []
                    else:
                        player_clicks = [selected_square]

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_z:  # when z is press, perform undo action
                    board.undo_move()
                    selected_square = ()
                    player_clicks = []

        draw_game_sate(screen, board)
        clock.tick(MAX_FPS)
        pygame.display.flip()


def draw_game_sate(screen, board):
    """
    Responsible for drawing current game state
    :param screen: the current screen to draw to
    :param board: the current game board to draw
    :return: void
    """
    draw_board(screen)
    draw_pieces(screen, board)


def draw_board(screen):
    """
    Draw the board on the screen without any pieces
    :param screen: the screen to draw to
    :return: void
    """
    # top left square is always white
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            # white tiles will always have an even number
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_pieces(screen, board):
    """
    Draw pieces from the board on the screen
    :param screen: the screen to draw on
    :param board: the current board
    :return:
    """
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board.piece_at([r, c])

            if piece.full_name() != "--":
                screen.blit(IMAGES[piece.full_name()], pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


# this is proper pyhon initialization to make sure the main method is called
if __name__ == "__main__":
    main()
