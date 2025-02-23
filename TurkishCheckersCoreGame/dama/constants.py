import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH//COLS

LIGHT_BROWN = (238, 220, 151)
DARK_BROWN = (150, 77, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (120, 120, 120)

CROWN = pygame.transform.scale(pygame.image.load('assets/crown.png'), (44, 25))