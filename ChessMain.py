"""
This is the main driver file and is responsible for handling user input and displaying current game state

see https://www.youtube.com/watch?v=EnYui0e73Rs&list=PLBwF487qi8MGU81nDGaeNE1EnNEPYWKY_&ab_channel=EddieSharick for some
the guide this is derived from.  The images for the chess pieces are also sourced from the video series description.
"""

import pygame

from board import Board
from chessEngine import GameState
from move import Move

WIDTH = HEIGHT = 512
DIMENSION = 8  # boards are 8 by 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15  # for animations later on
IMAGES = {}


def load_images():
    """
    Initialize global dictionary of images, called exactly once in main
    """
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))


def main():
    """
    Main driver for the game.  handles user input and updating graphics
    :return: void
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(pygame.Color("white"))
    game_state = GameState()
    board = Board()
    load_images()  # only do this once
    running = True

    valid_moves = game_state.get_valid_moves()
    move_made = False

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
                    move = Move.from_clicks(player_clicks[0], player_clicks[1], game_state.board)
                    print(move.get_chess_notation())

                    # according to the video i am following, when it comes to more complex moves like castling or en
                    # passant, we can't just use the old comparison of move in valid_moves because that will cause bugs
                    # instead we need to go through all of the valid moves generated by the engine, and specifically
                    # execute the move from the engine which matches the user input.  Of course, the risk here is
                    # potentially when we have multiple matching moves, or spending more time iterating through the
                    # list of moves than needed (we could always create an early exit by breaking out of the look as
                    # needed)
                    for valid_move in filter(lambda v_move: v_move == move, valid_moves):
                        game_state.make_move(valid_move)
                        print(f'board made successful move {board.make_move(move)}')
                        move_made = True
                        selected_square = ()
                        player_clicks = []

                    if not move_made:
                        player_clicks = [selected_square]

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_z:  # when z is press, perform undo action
                    game_state.undo_move()
                    board.undo_move()
                    move_made = True

        draw_game_sate(screen, game_state)
        clock.tick(MAX_FPS)
        pygame.display.flip()

        if move_made:
            valid_moves = game_state.get_valid_moves()
            move_made = False


def draw_game_sate(screen, game_sate):
    """
    Responsible for drawing current game state
    :param screen: the current screen to draw to
    :param game_sate: the current game state to draw from
    :return: void
    """
    draw_board(screen)
    draw_pieces(screen, game_sate.board)


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
    :param board: the current game_state.board
    :return:
    """
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]

            if piece != "--":
                screen.blit(IMAGES[piece], pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


# this is proper pyhon initialization to make sure the main method is called
if __name__ == "__main__":
    main()
