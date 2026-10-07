import pygame

# Constants
SQ_WIDTH = 50
UI_WIDTH = 100

SQ_LOC = []
for i in range(1, 9):
    for j in range(1, 10):
        SQ_LOC.append((SQ_WIDTH * i + i, SQ_WIDTH * j + j))
DIAG_LOC = [[(SQ_WIDTH * 4 + 3, SQ_WIDTH * 1 + 0), (SQ_WIDTH * 6 + 5, SQ_WIDTH *  3 + 2)], 
            [(SQ_WIDTH * 6 + 5, SQ_WIDTH * 1 + 0), (SQ_WIDTH * 4 + 3, SQ_WIDTH *  3 + 2)], 
            [(SQ_WIDTH * 4 + 3, SQ_WIDTH * 8 + 7), (SQ_WIDTH * 6 + 5, SQ_WIDTH * 10 + 9)], 
            [(SQ_WIDTH * 6 + 5, SQ_WIDTH * 8 + 7), (SQ_WIDTH * 4 + 3, SQ_WIDTH * 10 + 9)]]
MARK_LOC = [[(SQ_WIDTH * 2 + 1, SQ_WIDTH * 3 + 2), 0], [(SQ_WIDTH * 8 + 7, SQ_WIDTH * 3 + 2), 0],
            [(SQ_WIDTH * 1 + 0, SQ_WIDTH * 4 + 3), 2], [(SQ_WIDTH * 3 + 2, SQ_WIDTH * 4 + 3), 0],
            [(SQ_WIDTH * 5 + 4, SQ_WIDTH * 4 + 3), 0], [(SQ_WIDTH * 7 + 6, SQ_WIDTH * 4 + 3), 0], 
            [(SQ_WIDTH * 9 + 8, SQ_WIDTH * 4 + 3), 1], [(SQ_WIDTH * 1 + 0, SQ_WIDTH * 7 + 6), 2], 
            [(SQ_WIDTH * 3 + 2, SQ_WIDTH * 7 + 6), 0], [(SQ_WIDTH * 5 + 4, SQ_WIDTH * 7 + 6), 0], 
            [(SQ_WIDTH * 7 + 6, SQ_WIDTH * 7 + 6), 0], [(SQ_WIDTH * 9 + 8, SQ_WIDTH * 7 + 6), 1],
            [(SQ_WIDTH * 2 + 1, SQ_WIDTH * 8 + 7), 0], [(SQ_WIDTH * 8 + 7, SQ_WIDTH * 8 + 7), 0]]

PIECE_LOC_NAME = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9",
                  "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                  "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
                  "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9",
                  "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                  "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
                  "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
                  "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
                  "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

CORD_DICT = {}
for j in range(10):
    for i in range(9):
        x = SQ_WIDTH * (i+1) + i
        y = SQ_WIDTH * (j+1) + j
        index = j * 9 + i
        CORD_DICT[PIECE_LOC_NAME[index]] = (x, y)

# Colors
CLR_GameBg  = (255, 255, 255)
CLR_BoardBg = (206,  92,   0)
CLR_BoardSq = (252, 175,  62)
CLR_Outline = (  0,   0,   0)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((SQ_WIDTH * 10 + 9 + UI_WIDTH, SQ_WIDTH * 11 + 10))
clock = pygame.time.Clock()
running = True

# Pieces
B_General  = pygame.image.load("./asset/B_General.png").convert_alpha()
B_Advisor  = pygame.image.load("./asset/B_Advisor.png").convert_alpha()
B_Elephant = pygame.image.load("./asset/B_Elephant.png").convert_alpha()
B_Horse    = pygame.image.load("./asset/B_Horse.png").convert_alpha()
B_Chariot  = pygame.image.load("./asset/B_Chariot.png").convert_alpha()
B_Cannon   = pygame.image.load("./asset/B_Cannon.png").convert_alpha()
B_Soldier  = pygame.image.load("./asset/B_Soldier.png").convert_alpha()

R_General  = pygame.image.load("./asset/R_General.png").convert_alpha()
R_Advisor  = pygame.image.load("./asset/R_Advisor.png").convert_alpha()
R_Elephant = pygame.image.load("./asset/R_Elephant.png").convert_alpha()
R_Horse    = pygame.image.load("./asset/R_Horse.png").convert_alpha()
R_Chariot  = pygame.image.load("./asset/R_Chariot.png").convert_alpha()
R_Cannon   = pygame.image.load("./asset/R_Cannon.png").convert_alpha()
R_Soldier  = pygame.image.load("./asset/R_Soldier.png").convert_alpha()

def render_board(screen):
    pygame.draw.rect(screen, CLR_BoardBg, (0, 0, SQ_WIDTH * 10 + 9, SQ_WIDTH * 11 + 10))              # Board background
    pygame.draw.rect(screen, CLR_Outline, (SQ_WIDTH, SQ_WIDTH, SQ_WIDTH * 8 + 9, SQ_WIDTH * 9 + 10))  # Board outline
    for loc in SQ_LOC:                                                                                # Squares
        pygame.draw.rect(screen, CLR_BoardSq, (loc[0], loc[1], SQ_WIDTH, SQ_WIDTH))
    pygame.draw.rect(screen, CLR_BoardSq, (SQ_LOC[4][0], SQ_LOC[4][1], SQ_WIDTH * 8 + 7, SQ_WIDTH))   # River
    for loc in DIAG_LOC:                                                                              # Palaces
        pygame.draw.line(screen, CLR_Outline, loc[0], loc[1], width=1)
    for loc, t in MARK_LOC:                                                                           # Marks
        if t == 0 or t == 1:
            pygame.draw.line(screen, CLR_Outline, (loc[0] - 10, loc[1] -  3), (loc[0] - 3, loc[1] - 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] -  3, loc[1] - 10), (loc[0] - 3, loc[1] - 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] - 10, loc[1] +  3), (loc[0] - 3, loc[1] + 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] -  3, loc[1] + 10), (loc[0] - 3, loc[1] + 3), width=1)
        if t == 0 or t == 2:
            pygame.draw.line(screen, CLR_Outline, (loc[0] + 10, loc[1] -  3), (loc[0] + 3, loc[1] - 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] +  3, loc[1] - 10), (loc[0] + 3, loc[1] - 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] + 10, loc[1] +  3), (loc[0] + 3, loc[1] + 3), width=1)
            pygame.draw.line(screen, CLR_Outline, (loc[0] +  3, loc[1] + 10), (loc[0] + 3, loc[1] + 3), width=1)

def render_piece(screen, piece, c):
    piece_width = int(SQ_WIDTH * 0.8) if int(SQ_WIDTH * 0.8) % 2 == 1 else int(SQ_WIDTH * 0.8) + 1
    pygame.draw.ellipse(screen, CLR_Outline, (c[0] - piece_width // 2, c[1] - piece_width // 2, piece_width, piece_width))
    screen.blit(pygame.transform.smoothscale(piece, (piece_width - 2, piece_width - 2)), (c[0] - piece_width // 2 + 1, c[1] - piece_width // 2 + 1))

while running:
    # Get events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw screen
    screen.fill(CLR_GameBg)
    render_board(screen)

    # Temp hardcoded placements
    render_piece(screen, R_Chariot , CORD_DICT["J1"])
    render_piece(screen, R_Horse   , CORD_DICT["J2"])
    render_piece(screen, R_Elephant, CORD_DICT["J3"])
    render_piece(screen, R_Advisor , CORD_DICT["J4"])
    render_piece(screen, R_General , CORD_DICT["J5"])
    render_piece(screen, R_Advisor , CORD_DICT["J6"])
    render_piece(screen, R_Elephant, CORD_DICT["J7"])
    render_piece(screen, R_Horse   , CORD_DICT["J8"])
    render_piece(screen, R_Chariot , CORD_DICT["J9"])

    render_piece(screen, R_Cannon  , CORD_DICT["H2"])
    render_piece(screen, R_Cannon  , CORD_DICT["H8"])

    render_piece(screen, R_Soldier , CORD_DICT["G1"])
    render_piece(screen, R_Soldier , CORD_DICT["G3"])
    render_piece(screen, R_Soldier , CORD_DICT["G5"])
    render_piece(screen, R_Soldier , CORD_DICT["G7"])
    render_piece(screen, R_Soldier , CORD_DICT["G9"])

    render_piece(screen, B_Soldier , CORD_DICT["D1"])
    render_piece(screen, B_Soldier , CORD_DICT["D3"])
    render_piece(screen, B_Soldier , CORD_DICT["D5"])
    render_piece(screen, B_Soldier , CORD_DICT["D7"])
    render_piece(screen, B_Soldier , CORD_DICT["D9"])

    render_piece(screen, B_Cannon  , CORD_DICT["C2"])
    render_piece(screen, B_Cannon  , CORD_DICT["C8"])

    render_piece(screen, B_Chariot , CORD_DICT["A1"])
    render_piece(screen, B_Horse   , CORD_DICT["A2"])
    render_piece(screen, B_Elephant, CORD_DICT["A3"])
    render_piece(screen, B_Advisor , CORD_DICT["A4"])
    render_piece(screen, B_General , CORD_DICT["A5"])
    render_piece(screen, B_Advisor , CORD_DICT["A6"])
    render_piece(screen, B_Elephant, CORD_DICT["A7"])
    render_piece(screen, B_Horse   , CORD_DICT["A8"])
    render_piece(screen, B_Chariot , CORD_DICT["A9"])

    pygame.display.flip()

    clock.tick(60)

pygame.quit()