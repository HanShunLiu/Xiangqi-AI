import pygame
import math

# Constants
SQ_WIDTH = 50
UI_WIDTH = 100

ROW = ["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]
COL = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
POS_TBL = {}
for r in range(10):
    for c in range(9):
        POS_TBL[ROW[r]+COL[c]] = (SQ_WIDTH * (c+1) + c, SQ_WIDTH * (r+1) + r)

# States
BOARD_STATE = {
    "RR1" : "J1",  # Red Chariot (Rook) 1
    "RH1" : "J2",  # Red Horse 1
    "RE1" : "J3",  # Red Elephant 1
    "RA1" : "J4",  # Red Advisor 1
    "RG_" : "J5",  # Red General
    "RA2" : "J6",  # Red Advisor 2
    "RE2" : "J7",  # Red Elephant 2
    "RH2" : "J8",  # Red Horse 2
    "RR2" : "J9",  # Red Chariot (Rook) 2

    "RC1" : "H2",  # Red Cannon 1
    "RC2" : "H8",  # Red Cannon 2

    "RS1" : "G1",  # Red Soldier 1
    "RS2" : "G3",  # Red Soldier 2
    "RS3" : "G5",  # Red Soldier 3
    "RS4" : "G7",  # Red Soldier 4
    "RS5" : "G9",  # Red Soldier 5

    "BS1" : "D1",  # Black Soldier 1
    "BS2" : "D3",  # Black Soldier 2
    "BS3" : "D5",  # Black Soldier 3
    "BS4" : "D7",  # Black Soldier 4
    "BS5" : "D9",  # Black Soldier 5

    "BC1" : "C2",  # Black Cannon 1
    "BC2" : "C8",  # Black Cannon 2

    "BR1" : "A1",  # Black Chariot (Rook) 1
    "BH1" : "A2",  # Black Horse 1
    "BE1" : "A3",  # Black Elephant 1
    "BA1" : "A4",  # Black Advisor 1
    "BG_" : "A5",  # Black General
    "BA2" : "A6",  # Black Advisor 2
    "BE2" : "A7",  # Black Elephant 2
    "BH2" : "A8",  # Black Horse 2
    "BR2" : "A9",  # Black Chariot (Rook) 2
}

MOUSE_STATE = ""  # empty string : IDLE, non-empty string: SELECT

# Colors
CLR_GameBg  = (255, 255, 255)
CLR_BoardBg = (206,  92,   0)
CLR_BoardSq = (252, 175,  62)
CLR_Outline = (  0,   0,   0)

# Pygame Setup
pygame.init()
screen = pygame.display.set_mode((SQ_WIDTH * 10 + 9 + UI_WIDTH, SQ_WIDTH * 11 + 10))
clock = pygame.time.Clock()
running = True

# Load Assets
RG_PNG = pygame.image.load("./asset/R_General.png").convert_alpha()
RA_PNG = pygame.image.load("./asset/R_Advisor.png").convert_alpha()
RE_PNG = pygame.image.load("./asset/R_Elephant.png").convert_alpha()
RH_PNG = pygame.image.load("./asset/R_Horse.png").convert_alpha()
RR_PNG = pygame.image.load("./asset/R_Chariot.png").convert_alpha()
RC_PNG = pygame.image.load("./asset/R_Cannon.png").convert_alpha()
RS_PNG = pygame.image.load("./asset/R_Soldier.png").convert_alpha()

BG_PNG = pygame.image.load("./asset/B_General.png").convert_alpha()
BA_PNG = pygame.image.load("./asset/B_Advisor.png").convert_alpha()
BE_PNG = pygame.image.load("./asset/B_Elephant.png").convert_alpha()
BH_PNG = pygame.image.load("./asset/B_Horse.png").convert_alpha()
BR_PNG = pygame.image.load("./asset/B_Chariot.png").convert_alpha()
BC_PNG = pygame.image.load("./asset/B_Cannon.png").convert_alpha()
BS_PNG = pygame.image.load("./asset/B_Soldier.png").convert_alpha()

# IDK where to put these
piece_width = int(SQ_WIDTH * 0.8) if int(SQ_WIDTH * 0.8) % 2 == 1 else int(SQ_WIDTH * 0.8) + 1

# Renders board background
def render_board():
    # Board Background
    pygame.draw.rect(screen, CLR_BoardBg, (0, 0, SQ_WIDTH * 10 + 9, SQ_WIDTH * 11 + 10))

    # Board Outline
    pygame.draw.rect(screen, CLR_Outline, (SQ_WIDTH, SQ_WIDTH, SQ_WIDTH * 8 + 9, SQ_WIDTH * 9 + 10))

    # Grid
    for r in ["J", "I", "H", "G", "E", "D", "C", "B"]:
        for c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            pos = POS_TBL[r+c]
            pygame.draw.rect(screen, CLR_BoardSq, (pos[0]+1, pos[1]+1, SQ_WIDTH, SQ_WIDTH))
    
    # River
    pygame.draw.rect(screen, CLR_BoardSq, (POS_TBL["F1"][0]+1, POS_TBL["F1"][1]+1, SQ_WIDTH * 8 + 7, SQ_WIDTH))

    # Palace
    for pos in [[POS_TBL["J4"], POS_TBL["H6"]], [POS_TBL["J6"], POS_TBL["H4"]], 
                [POS_TBL["C4"], POS_TBL["A6"]], [POS_TBL["C6"], POS_TBL["A4"]]]:
        pygame.draw.line(screen, CLR_Outline, pos[0], pos[1], width=1)
    
    # Marks
    for pos in [POS_TBL["H2"], POS_TBL["H8"], POS_TBL["G1"], POS_TBL["G3"], POS_TBL["G5"], POS_TBL["G7"], POS_TBL["G9"],
                POS_TBL["C2"], POS_TBL["C8"], POS_TBL["D1"], POS_TBL["D3"], POS_TBL["D5"], POS_TBL["D7"], POS_TBL["D9"]]:
        x = SQ_WIDTH // 5
        y = SQ_WIDTH // 15
        if pos[0] > SQ_WIDTH:
            pygame.draw.line(screen, CLR_Outline, (pos[0] - x, pos[1] - y), (pos[0] - y, pos[1] - y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] - y, pos[1] - x), (pos[0] - y, pos[1] - y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] - x, pos[1] + y), (pos[0] - y, pos[1] + y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] - y, pos[1] + x), (pos[0] - y, pos[1] + y), width=1)
        if pos[0] < SQ_WIDTH * 9 + 8:
            pygame.draw.line(screen, CLR_Outline, (pos[0] + x, pos[1] - y), (pos[0] + y, pos[1] - y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] + y, pos[1] - x), (pos[0] + y, pos[1] - y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] + x, pos[1] + y), (pos[0] + y, pos[1] + y), width=1)
            pygame.draw.line(screen, CLR_Outline, (pos[0] + y, pos[1] + x), (pos[0] + y, pos[1] + y), width=1)

# Renders all pieces based on board state
def render_pieces():
    for piece in BOARD_STATE:
        pos_key = BOARD_STATE[piece]
        if pos_key != "":
            pos = POS_TBL[pos_key]
            pygame.draw.ellipse(screen, CLR_Outline, (pos[0] - piece_width // 2, pos[1] - piece_width // 2, piece_width, piece_width))
            screen.blit(
                pygame.transform.smoothscale(globals()[piece[0:2] + "_PNG"], (piece_width - 2, piece_width - 2)), 
                (pos[0] - piece_width // 2 + 1, pos[1] - piece_width // 2 + 1)
            )

def move_piece(piece, pos):
    for p in BOARD_STATE:
        if p != piece and BOARD_STATE[p] == pos:
            BOARD_STATE[p] = ""
            break
    BOARD_STATE[piece] = pos

while running:
    # Get events
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            if MOUSE_STATE == "":
                for p in BOARD_STATE:
                    if BOARD_STATE[p] != "":
                        pos = POS_TBL[BOARD_STATE[p]]
                        if math.hypot(mouse_x - pos[0], mouse_y - pos[1]) <= piece_width // 2:
                            MOUSE_STATE = p
                            break
            else:
                for pos_name in POS_TBL:
                    pos = POS_TBL[pos_name]
                    if math.hypot(mouse_x - pos[0], mouse_y - pos[1]) <= piece_width // 2:
                        move_piece(MOUSE_STATE, pos_name)
                        MOUSE_STATE = ""
                        break
        
        if event.type == pygame.QUIT:
            running = False

    # Render screen
    screen.fill(CLR_GameBg)
    render_board()
    render_pieces()

    # Refresh screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
