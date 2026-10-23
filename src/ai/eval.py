from dataclasses import dataclass
from typing import Callable, Dict, List
from src.ai.board import BLACK, RED, XiangqiBoard

# Baseline piece values (in centipawns)
PIECE_VALUES = {
    "G": 10000,  # General
    "R": 900,    # Chariot (Rook)
    "C": 450,    # Cannon
    "H": 400,    # Horse
    "E": 200,    # Elephant
    "A": 200,    # Advisor
    "S": 100,    # Soldier (base value before crossing river)
}

CROSSED_RIVER_SOLDIER_BONUS = 100
MAX_NON_GENERAL_PIECES = 30

# Positional bonuses
RIVER_CONTROL_BONUS = {
    "R": 25,
    "C": 15,
    "H": 20,
    "S": 15,
}
PALACE_INFILTRATION_BONUS = {
    "R": 60,
    "H": 50,
    "C": 35,
    "S": 45,
}
NEAR_PALACE_BONUS = {
    "R": 25,
    "H": 25,
    "C": 15,
    "S": 20,
}

# 10x9 Piece-Square Tables from Red's perspective:
# Row 0 ('J') is Red's back rank; Row 9 ('A') is Black's back rank.
# All rows are left-right symmetric so the initial board evaluates to 0.
PST_SOLDIER: List[List[int]] = [
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # J (0)
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # I (1)
    [0,  0,  0,  0,  0,  0,  0,  0,  0],  # H (2)
    [0,  0,  2,  0,  6,  0,  2,  0,  0],  # G (3) - starting soldier rank
    [4,  0,  8,  0, 12,  0,  8,  0,  4],  # F (4) - Red riverbank
    [10, 14, 20, 28, 32, 28, 20, 14, 10], # E (5) - across river
    [16, 24, 32, 42, 48, 42, 32, 24, 16], # D (6)
    [20, 30, 42, 55, 62, 55, 42, 30, 20], # C (7) - Black palace top
    [20, 32, 45, 60, 68, 60, 45, 32, 20], # B (8) - Black palace center
    [12, 18, 25, 35, 40, 35, 25, 18, 12], # A (9) - Black back rank (loses forward step)
]

PST_HORSE: List[List[int]] = [
    [ 0, -4,  2,  2,  0,  2,  2, -4,  0],  # J (0)
    [ 2,  4,  8, 10,  8, 10,  8,  4,  2],  # I (1)
    [ 4,  8, 14, 16, 18, 16, 14,  8,  4],  # H (2)
    [ 6, 12, 18, 22, 24, 22, 18, 12,  6],  # G (3)
    [ 8, 14, 22, 26, 28, 26, 22, 14,  8],  # F (4)
    [10, 18, 26, 32, 34, 32, 26, 18, 10],  # E (5)
    [12, 22, 30, 38, 40, 38, 30, 22, 12],  # D (6)
    [10, 24, 34, 44, 42, 44, 34, 24, 10],  # C (7)
    [ 6, 18, 28, 38, 30, 38, 28, 18,  6],  # B (8)
    [ 0,  8, 14, 20, 14, 20, 14,  8,  0],  # A (9)
]

PST_CHARIOT: List[List[int]] = [
    [ -4,  4,  2,  8, 10,  8,  2,  4, -4],  # J (0)
    [  4,  8,  8, 14, 16, 14,  8,  8,  4],  # I (1)
    [  4, 10, 10, 16, 18, 16, 10, 10,  4],  # H (2)
    [  8, 14, 14, 22, 24, 22, 14, 14,  8],  # G (3)
    [ 12, 18, 18, 26, 28, 26, 18, 18, 12],  # F (4)
    [ 14, 22, 22, 30, 32, 30, 22, 22, 14],  # E (5)
    [ 18, 26, 26, 34, 36, 34, 26, 26, 18],  # D (6)
    [ 18, 26, 28, 38, 40, 38, 28, 26, 18],  # C (7)
    [ 20, 28, 30, 42, 45, 42, 30, 28, 20],  # B (8)
    [ 14, 20, 22, 32, 36, 32, 22, 20, 14],  # A (9)
]

PST_CANNON: List[List[int]] = [
    [ 0,  0,  2,  6,  8,  6,  2,  0,  0],  # J (0)
    [ 0,  4,  6, 10, 14, 10,  6,  4,  0],  # I (1)
    [ 2,  6,  8, 14, 22, 14,  8,  6,  2],  # H (2)
    [ 0,  4,  6, 14, 24, 14,  6,  4,  0],  # G (3)
    [ 0,  4,  8, 16, 26, 16,  8,  4,  0],  # F (4)
    [ 2,  6, 10, 18, 28, 18, 10,  6,  2],  # E (5)
    [ 4,  8, 12, 20, 30, 20, 12,  8,  4],  # D (6)
    [ 6, 10, 14, 22, 32, 22, 14, 10,  6],  # C (7)
    [ 4,  8, 10, 16, 22, 16, 10,  8,  4],  # B (8)
    [ 2,  4,  6, 12, 18, 12,  6,  4,  2],  # A (9)
]

PST_TABLES: Dict[str, List[List[int]]] = {
    "S": PST_SOLDIER,
    "H": PST_HORSE,
    "R": PST_CHARIOT,
    "C": PST_CANNON,
}


@dataclass(frozen=True)
class EvalWeights:
    """
    Configurable scaling weights for each heuristic evaluation component,
    supporting ablation experiments and parameter tuning.
    """
    dynamic_phase: float = 1.0
    pst: float = 1.0
    river_control: float = 1.0
    palace_infiltration: float = 1.0
    piece_activity: float = 1.0
    king_safety: float = 1.0


DEFAULT_WEIGHTS = EvalWeights()


def get_pst_bonus(piece: str, r: int, c: int) -> int:
    """Return the Piece-Square Table bonus for `piece` at (r, c)."""
    ptype = piece[1]
    table = PST_TABLES.get(ptype)
    if table is None:
        return 0
    lookup_r = r if piece[0] == RED else (9 - r)
    return table[lookup_r][c]


def get_piece_value(piece: str, r: int, c: int) -> int:
    """Return baseline static material value of a single piece at (r, c)."""
    color = piece[0]
    ptype = piece[1]
    val = PIECE_VALUES[ptype]
    if ptype == "S" and not XiangqiBoard.on_own_side(r, color):
        val += CROSSED_RIVER_SOLDIER_BONUS
    return val


def evaluate_material(board: XiangqiBoard, perspective: str = RED) -> int:
    """
    Compute pure baseline material evaluation (plus crossed-river soldier bonus)
    from `perspective`'s point of view.
    """
    red_score = 0
    black_score = 0

    for piece, rc in board.piece_coords.items():
        if rc is None:
            continue
        r, c = rc
        val = get_piece_value(piece, r, c)
        if piece[0] == RED:
            red_score += val
        else:
            black_score += val

    score = red_score - black_score
    return score if perspective == RED else -score


def _game_phase_crowding(board: XiangqiBoard) -> float:
    """
    Return a crowding ratio in [0.0, 1.0] based on remaining non-General pieces.
    1.0 = full opening board (30 non-General pieces), 0.0 = bare Generals.
    """
    count = 0
    for piece, rc in board.piece_coords.items():
        if rc is not None and piece[1] != "G":
            count += 1
    return min(1.0, count / MAX_NON_GENERAL_PIECES)


def _dynamic_material_value(
    piece: str,
    r: int,
    c: int,
    crowding: float,
    phase_weight: float = 1.0,
) -> int:
    """
    Compute dynamic material weight adjusted for game stage:
    - Cannons are worth more in crowded openings (more screens) and less in endgames.
    - Horses and crossed-river Soldiers gain value as the board opens up in endgames.
    """
    color = piece[0]
    ptype = piece[1]
    val = PIECE_VALUES[ptype]

    if ptype == "C":
        val += int(phase_weight * 60 * (crowding - 0.5))
    elif ptype == "H":
        val += int(phase_weight * 60 * (0.5 - crowding))
    elif ptype == "S":
        if not XiangqiBoard.on_own_side(r, color):
            val += CROSSED_RIVER_SOLDIER_BONUS
            val += int(phase_weight * 30 * (1.0 - crowding))

    return val


def _piece_activity_and_space(
    board: XiangqiBoard,
    piece: str,
    r: int,
    c: int,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> int:
    """
    Evaluate PST, piece activity, river control, and opponent palace infiltration.
    """
    color = piece[0]
    ptype = piece[1]
    opp_color = BLACK if color == RED else RED
    bonus = 0.0

    # 0. Piece-Square Table (PST) bonus
    if weights.pst != 0.0:
        bonus += weights.pst * get_pst_bonus(piece, r, c)

    # 1. Control of the river (rows 4 and 5)
    if weights.river_control != 0.0:
        if r in (4, 5):
            bonus += weights.river_control * RIVER_CONTROL_BONUS.get(ptype, 0)
        elif (color == RED and r == 6) or (color == BLACK and r == 3):
            bonus += weights.river_control * (RIVER_CONTROL_BONUS.get(ptype, 0) // 2)

    # 2. Palace infiltration (inside opponent's 3x3 palace or 1 step outside)
    if weights.palace_infiltration != 0.0:
        if XiangqiBoard.in_palace(r, c, opp_color):
            bonus += weights.palace_infiltration * PALACE_INFILTRATION_BONUS.get(ptype, 0)
        else:
            opp_palace_rows = (7, 8, 9) if color == RED else (0, 1, 2)
            if r in opp_palace_rows and 2 <= c <= 6:
                bonus += weights.palace_infiltration * NEAR_PALACE_BONUS.get(ptype, 0)

    # 3. Piece-specific activity & mobility heuristics
    if weights.piece_activity != 0.0:
        act = 0
        if ptype == "H":
            open_legs = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                lr, lc = r + dr, c + dc
                if XiangqiBoard.in_bounds(lr, lc) and board.grid[lr][lc] is None:
                    open_legs += 1
            act += (open_legs - 2) * 10
            if 2 <= c <= 6:
                act += 10
        elif ptype == "R":
            open_rays = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if XiangqiBoard.in_bounds(nr, nc) and board.grid[nr][nc] is None:
                    open_rays += 1
            act += open_rays * 6
            if 3 <= c <= 5:
                act += 10
        elif ptype == "C":
            if c == 4:
                act += 20
        elif ptype == "S":
            if not XiangqiBoard.on_own_side(r, color):
                advance = r - 4 if color == RED else 5 - r
                act += advance * 8
                if 3 <= c <= 5:
                    act += 12
        bonus += weights.piece_activity * act

    return int(bonus)


def _king_safety(board: XiangqiBoard, color: str) -> int:
    """
    Evaluate King (General) safety for `color`:
    - Bonus for surviving Advisors and Elephants guarding the side.
    - Penalty for Flying General thin screens (only 1 piece between Generals on same file).
    - Penalty for enemy Chariots or Cannons aligned on the General's file/rank.
    """
    gen_key = "RG_" if color == RED else "BG_"
    gen_rc = board.piece_coords.get(gen_key)
    if gen_rc is None:
        return -5000

    gr, gc = gen_rc
    safety = 0

    advisors = 0
    elephants = 0
    for p, rc in board.piece_coords.items():
        if rc is not None and p[0] == color:
            if p[1] == "A":
                advisors += 1
            elif p[1] == "E":
                elephants += 1
    safety += advisors * 25 + elephants * 15
    if advisors == 2 and elephants == 2:
        safety += 20

    opp_gen_key = "BG_" if color == RED else "RG_"
    opp_gen_rc = board.piece_coords.get(opp_gen_key)
    if opp_gen_rc is not None and opp_gen_rc[1] == gc:
        low, high = (gr, opp_gen_rc[0]) if gr < opp_gen_rc[0] else (opp_gen_rc[0], gr)
        screens = sum(1 for r in range(low + 1, high) if board.grid[r][gc] is not None)
        if screens <= 1:
            safety -= 35

    opp_color = BLACK if color == RED else RED
    for p, rc in board.piece_coords.items():
        if rc is not None and p[0] == opp_color and p[1] in ("R", "C"):
            pr, pc = rc
            if pr == gr or pc == gc:
                safety -= 25

    return safety


def evaluate_board(
    board: XiangqiBoard,
    perspective: str = RED,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> int:
    """
    Full heuristic evaluation combining:
    - Dynamic material weighting (game-phase adjusted)
    - Piece-Square Tables (PST)
    - Piece activity & mobility
    - Control of the river
    - Palace infiltration
    - King safety
    Positive scores favor `perspective`; negative scores favor the opponent.
    """
    crowding = _game_phase_crowding(board)
    red_score = 0
    black_score = 0

    for piece, rc in board.piece_coords.items():
        if rc is None:
            continue
        r, c = rc
        val = _dynamic_material_value(piece, r, c, crowding, weights.dynamic_phase)
        val += _piece_activity_and_space(board, piece, r, c, weights)
        if piece[0] == RED:
            red_score += val
        else:
            black_score += val

    if weights.king_safety != 0.0:
        red_score += int(weights.king_safety * _king_safety(board, RED))
        black_score += int(weights.king_safety * _king_safety(board, BLACK))

    total = red_score - black_score
    return total if perspective == RED else -total


def make_weighted_evaluator(weights: EvalWeights) -> Callable[[XiangqiBoard, str], int]:
    """Create a custom `eval_fn(board, perspective)` with specified `EvalWeights`."""
    def _eval(board: XiangqiBoard, perspective: str = RED) -> int:
        return evaluate_board(board, perspective=perspective, weights=weights)
    return _eval