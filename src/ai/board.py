from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Coordinate mappings matching src/gui/game.py
ROWS = ["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]
COLS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

ROW_TO_IDX = {r: i for i, r in enumerate(ROWS)}
COL_TO_IDX = {c: i for i, c in enumerate(COLS)}

RED = "R"
BLACK = "B"

INITIAL_BOARD_STATE: Dict[str, str] = {
    "RR1": "J1", "RH1": "J2", "RE1": "J3", "RA1": "J4", "RG_": "J5",
    "RA2": "J6", "RE2": "J7", "RH2": "J8", "RR2": "J9",
    "RC1": "H2", "RC2": "H8",
    "RS1": "G1", "RS2": "G3", "RS3": "G5", "RS4": "G7", "RS5": "G9",
    "BS1": "D1", "BS2": "D3", "BS3": "D5", "BS4": "D7", "BS5": "D9",
    "BC1": "C2", "BC2": "C8",
    "BR1": "A1", "BH1": "A2", "BE1": "A3", "BA1": "A4", "BG_": "A5",
    "BA2": "A6", "BE2": "A7", "BH2": "A8", "BR2": "A9",
}


def pos_to_coords(pos: str) -> Tuple[int, int]:
    """Convert GUI position string (e.g., 'J1') to (row, col) indices."""
    return ROW_TO_IDX[pos[0]], COL_TO_IDX[pos[1]]


def coords_to_pos(r: int, c: int) -> str:
    """Convert (row, col) indices to GUI position string (e.g., 'J1')."""
    return ROWS[r] + COLS[c]


@dataclass(frozen=True)
class Move:
    piece: str
    from_rc: Tuple[int, int]
    to_rc: Tuple[int, int]
    captured: Optional[str] = None

    @property
    def from_pos(self) -> str:
        return coords_to_pos(*self.from_rc)

    @property
    def to_pos(self) -> str:
        return coords_to_pos(*self.to_rc)

    def __repr__(self) -> str:
        cap = f"x{self.captured}" if self.captured else ""
        return f"{self.piece}:{self.from_pos}->{self.to_pos}{cap}"


class XiangqiBoard:
    """
    Fast 10x9 board representation compatible with game.py's BOARD_STATE.
    Red occupies rows 0..4 (J..F) initially and moves down (increasing row index).
    Black occupies rows 5..9 (E..A) initially and moves up (decreasing row index).
    """

    def __init__(self, board_state: Optional[Dict[str, str]] = None, side_to_move: str = RED):
        self.grid: List[List[Optional[str]]] = [[None] * 9 for _ in range(10)]
        self.piece_coords: Dict[str, Optional[Tuple[int, int]]] = {}
        self.side_to_move: str = side_to_move
        self.move_history: List[Move] = []

        state = board_state if board_state is not None else INITIAL_BOARD_STATE
        self.load_board_state(state)

    def load_board_state(self, board_state: Dict[str, str]) -> None:
        """Populate internal grid and piece tables from a game.py BOARD_STATE dict."""
        self.grid = [[None] * 9 for _ in range(10)]
        self.piece_coords = {}
        for piece, pos_str in board_state.items():
            if pos_str and len(pos_str) == 2:
                r, c = pos_to_coords(pos_str)
                self.grid[r][c] = piece
                self.piece_coords[piece] = (r, c)
            else:
                self.piece_coords[piece] = None

    def to_board_state(self) -> Dict[str, str]:
        """Export current board to a dictionary matching game.py's BOARD_STATE format."""
        state: Dict[str, str] = {}
        for piece, rc in self.piece_coords.items():
            state[piece] = coords_to_pos(*rc) if rc is not None else ""
        return state

    @staticmethod
    def opponent(color: str) -> str:
        return BLACK if color == RED else RED

    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < 10 and 0 <= c < 9

    @staticmethod
    def in_palace(r: int, c: int, color: str) -> bool:
        if not (3 <= c <= 5):
            return False
        return (0 <= r <= 2) if color == RED else (7 <= r <= 9)

    @staticmethod
    def on_own_side(r: int, color: str) -> bool:
        return (0 <= r <= 4) if color == RED else (5 <= r <= 9)

    def make_move(self, move: Move) -> None:
        """Apply a move in-place and push it to move_history."""
        fr, fc = move.from_rc
        tr, tc = move.to_rc

        if move.captured is not None:
            self.piece_coords[move.captured] = None

        self.grid[fr][fc] = None
        self.grid[tr][tc] = move.piece
        self.piece_coords[move.piece] = (tr, tc)
        self.move_history.append(move)
        self.side_to_move = self.opponent(self.side_to_move)

    def undo_move(self) -> Move:
        """Revert the most recent move in-place."""
        move = self.move_history.pop()
        fr, fc = move.from_rc
        tr, tc = move.to_rc

        self.grid[fr][fc] = move.piece
        self.piece_coords[move.piece] = (fr, fc)

        if move.captured is not None:
            self.grid[tr][tc] = move.captured
            self.piece_coords[move.captured] = (tr, tc)
        else:
            self.grid[tr][tc] = None

        self.side_to_move = self.opponent(self.side_to_move)
        return move

    # -------------------------------------------------------------------------
    # Pseudo-Legal Move Generation
    # -------------------------------------------------------------------------
    def get_pseudo_legal_moves(self, color: Optional[str] = None) -> List[Move]:
        """Generate all pseudo-legal moves for `color` without checking king safety."""
        color = color or self.side_to_move
        moves: List[Move] = []

        for piece, rc in self.piece_coords.items():
            if rc is None or piece[0] != color:
                continue
            ptype = piece[1]
            r, c = rc
            if ptype == "G":
                self._gen_general_moves(piece, r, c, color, moves)
            elif ptype == "A":
                self._gen_advisor_moves(piece, r, c, color, moves)
            elif ptype == "E":
                self._gen_elephant_moves(piece, r, c, color, moves)
            elif ptype == "H":
                self._gen_horse_moves(piece, r, c, color, moves)
            elif ptype == "R":
                self._gen_chariot_moves(piece, r, c, color, moves)
            elif ptype == "C":
                self._gen_cannon_moves(piece, r, c, color, moves)
            elif ptype == "S":
                self._gen_soldier_moves(piece, r, c, color, moves)

        return moves

    def _add_target(self, piece: str, r: int, c: int, tr: int, tc: int, color: str, moves: List[Move]) -> None:
        target = self.grid[tr][tc]
        if target is None or target[0] != color:
            moves.append(Move(piece, (r, c), (tr, tc), target))

    def _gen_general_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = r + dr, c + dc
            if self.in_palace(tr, tc, color):
                self._add_target(piece, r, c, tr, tc, color, moves)

    def _gen_advisor_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            tr, tc = r + dr, c + dc
            if self.in_palace(tr, tc, color):
                self._add_target(piece, r, c, tr, tc, color, moves)

    def _gen_elephant_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            tr, tc = r + dr, c + dc
            eye_r, eye_c = r + dr // 2, c + dc // 2
            if self.in_bounds(tr, tc) and self.on_own_side(tr, color):
                if self.grid[eye_r][eye_c] is None:  # Elephant eye must be unblocked
                    self._add_target(piece, r, c, tr, tc, color, moves)

    def _gen_horse_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        horse_steps = (
            ((-1, 0), (-2, -1)), ((-1, 0), (-2, 1)),
            ((1, 0), (2, -1)),   ((1, 0), (2, 1)),
            ((0, -1), (-1, -2)), ((0, -1), (1, -2)),
            ((0, 1), (-1, 2)),   ((0, 1), (1, 2)),
        )
        for (leg_dr, leg_dc), (dr, dc) in horse_steps:
            leg_r, leg_c = r + leg_dr, c + leg_dc
            tr, tc = r + dr, c + dc
            if self.in_bounds(tr, tc) and self.grid[leg_r][leg_c] is None:
                self._add_target(piece, r, c, tr, tc, color, moves)

    def _gen_chariot_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = r + dr, c + dc
            while self.in_bounds(tr, tc):
                target = self.grid[tr][tc]
                if target is None:
                    moves.append(Move(piece, (r, c), (tr, tc), None))
                else:
                    if target[0] != color:
                        moves.append(Move(piece, (r, c), (tr, tc), target))
                    break
                tr += dr
                tc += dc

    def _gen_cannon_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = r + dr, c + dc
            jumped = False
            while self.in_bounds(tr, tc):
                target = self.grid[tr][tc]
                if not jumped:
                    if target is None:
                        moves.append(Move(piece, (r, c), (tr, tc), None))
                    else:
                        jumped = True
                else:
                    if target is not None:
                        if target[0] != color:
                            moves.append(Move(piece, (r, c), (tr, tc), target))
                        break
                tr += dr
                tc += dc

    def _gen_soldier_moves(self, piece: str, r: int, c: int, color: str, moves: List[Move]) -> None:
        forward_dr = 1 if color == RED else -1
        tr, tc = r + forward_dr, c
        if self.in_bounds(tr, tc):
            self._add_target(piece, r, c, tr, tc, color, moves)

        # Once across the river, soldiers can also move horizontally
        if not self.on_own_side(r, color):
            for dc in (-1, 1):
                tr, tc = r, c + dc
                if self.in_bounds(tr, tc):
                    self._add_target(piece, r, c, tr, tc, color, moves)

    # -------------------------------------------------------------------------
    # Check, Flying General & Legal Move Filtering
    # -------------------------------------------------------------------------
    def is_flying_general(self) -> bool:
        """Return True if both Generals face each other on the same column with no pieces between."""
        rg_rc = self.piece_coords.get("RG_")
        bg_rc = self.piece_coords.get("BG_")
        if rg_rc is None or bg_rc is None:
            return False
        rr, rc = rg_rc
        br, bc = bg_rc
        if rc != bc:
            return False
        low, high = (rr, br) if rr < br else (br, rr)
        for r in range(low + 1, high):
            if self.grid[r][rc] is not None:
                return False
        return True

    def is_in_check(self, color: str) -> bool:
        """
        Return True if `color`'s General is under attack or violating the Flying General rule.
        Uses fast direct ray and horse-leg inspection from the General's coordinates.
        """
        gen_key = "RG_" if color == RED else "BG_"
        gen_rc = self.piece_coords.get(gen_key)
        if gen_rc is None:
            return True

        gr, gc = gen_rc
        opp_color = self.opponent(color)
        opp_forward = 1 if opp_color == RED else -1

        # 1. Orthogonal rays: Chariot ('R'), Cannon ('C'), Flying General ('G'), and Soldier ('S')
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = gr + dr, gc + dc
            screens = 0
            while self.in_bounds(tr, tc):
                target = self.grid[tr][tc]
                if target is not None:
                    if screens == 0:
                        if target[0] == opp_color:
                            ptype = target[1]
                            if ptype == "R" or ptype == "G":
                                return True
                            if ptype == "S" and abs(tr - gr) + abs(tc - gc) == 1:
                                # Enemy soldier attacks forward, or sideways if across the river
                                if (gr - tr == opp_forward and tc == gc) or (
                                    tr == gr and not self.on_own_side(tr, opp_color)
                                ):
                                    return True
                        screens = 1
                    else:
                        if target[0] == opp_color and target[1] == "C":
                            return True
                        break
                tr += dr
                tc += dc

        # 2. Horse ('H') attacks: check all 8 knight-move offsets and their orthogonal leg
        horse_jumps = (
            (-2, -1), (-2, 1),
            (2, -1),  (2, 1),
            (-1, -2), (1, -2),
            (-1, 2),  (1, 2),
        )
        for dr, dc in horse_jumps:
            hr, hc = gr + dr, gc + dc
            if self.in_bounds(hr, hc):
                piece = self.grid[hr][hc]
                if piece is not None and piece[0] == opp_color and piece[1] == "H":
                    leg_r = gr + dr // 2
                    leg_c = gc + dc // 2
                    if self.grid[leg_r][leg_c] is None:
                        return True

        return False

    def get_legal_moves(self, color: Optional[str] = None) -> List[Move]:
        """Generate all strictly legal moves for `color` (filtering out moves that leave General in check)."""
        color = color or self.side_to_move
        legal_moves: List[Move] = []
        for move in self.get_pseudo_legal_moves(color):
            self.make_move(move)
            if not self.is_in_check(color):
                legal_moves.append(move)
            self.undo_move()
        return legal_moves

    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the current side_to_move has no legal moves.
        In Xiangqi, both checkmate and stalemate are losses for the side with no legal moves.
        Returns (is_over, winner_color).
        """
        if len(self.get_legal_moves(self.side_to_move)) == 0:
            return True, self.opponent(self.side_to_move)
        return False, None