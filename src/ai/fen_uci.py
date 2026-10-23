import subprocess
from typing import Dict, List, Optional, Tuple

from src.ai.board import BLACK, RED, Move, XiangqiBoard
from src.ai.search import SearchResult

# Standard Xiangqi FEN piece letters:
# Uppercase = Red (K/G=General, A=Advisor, B/E=Elephant, N/H=Horse, R=Chariot, C=Cannon, P/S=Soldier)
# Lowercase = Black
PTYPE_TO_FEN = {
    "G": "K",
    "A": "A",
    "E": "B",
    "H": "N",
    "R": "R",
    "C": "C",
    "S": "P",
}

FEN_TO_PTYPE = {
    "k": "G", "g": "G",
    "a": "A",
    "b": "E", "e": "E",
    "n": "H", "h": "H",
    "r": "R",
    "c": "C",
    "p": "S", "s": "S",
}

UCI_FILES = "abcdefghi"


def board_to_fen(board: XiangqiBoard, move_number: int = 1) -> str:
    """
    Convert a `XiangqiBoard` into a standard Xiangqi FEN string.
    In standard Xiangqi FEN, Black's back rank (row 9, 'A') is listed first (top rank 9)
    and Red's back rank (row 0, 'J') is listed last (bottom rank 0).
    """
    ranks_str: List[str] = []
    # Iterate from Black's back rank (r=9) down to Red's back rank (r=0)
    for r in range(9, -1, -1):
        empty = 0
        row_chars: List[str] = []
        for c in range(9):
            piece = board.grid[r][c]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    row_chars.append(str(empty))
                    empty = 0
                fen_char = PTYPE_TO_FEN[piece[1]]
                row_chars.append(fen_char if piece[0] == RED else fen_char.lower())
        if empty > 0:
            row_chars.append(str(empty))
        ranks_str.append("".join(row_chars))

    side_char = "w" if board.side_to_move == RED else "b"
    return f"{'/'.join(ranks_str)} {side_char} - - 0 {move_number}"


def fen_to_board(fen: str) -> XiangqiBoard:
    """
    Parse a standard Xiangqi FEN string and return a `XiangqiBoard` compatible
    with `game.py`'s piece naming scheme ('RR1', 'RG_', etc.).
    """
    parts = fen.strip().split()
    board_part = parts[0]
    side_char = parts[1].lower() if len(parts) > 1 else "w"
    side_to_move = RED if side_char in ("w", "r") else BLACK

    ranks = board_part.split("/")
    if len(ranks) != 10:
        raise ValueError(f"Expected 10 ranks in Xiangqi FEN, got {len(ranks)}")

    counters: Dict[str, int] = {}
    board_state: Dict[str, str] = {}

    # Rank 0 in FEN is Black's back rank (internal row 9, 'A');
    # Rank 9 in FEN is Red's back rank (internal row 0, 'J').
    from src.ai.board import coords_to_pos

    for fen_idx, rank_str in enumerate(ranks):
        r = 9 - fen_idx
        c = 0
        for ch in rank_str:
            if ch.isdigit():
                c += int(ch)
            else:
                color = RED if ch.isupper() else BLACK
                ptype = FEN_TO_PTYPE[ch.lower()]
                prefix = color + ptype
                if ptype == "G":
                    piece_key = f"{prefix}_"
                else:
                    counters[prefix] = counters.get(prefix, 0) + 1
                    piece_key = f"{prefix}{counters[prefix]}"
                board_state[piece_key] = coords_to_pos(r, c)
                c += 1

    return XiangqiBoard(board_state, side_to_move=side_to_move)


def move_to_uci(move: Move, zero_indexed_ranks: bool = True) -> str:
    """
    Convert an internal `Move` to UCI/UCCI coordinate notation (e.g., 'b2e2' or 'h2e2').
    In standard Xiangqi UCI/UCCI:
    - Files are 'a'..'i' (columns 0..8, matching game.py's '1'..'9').
    - Red's back rank (row 0, 'J') is rank 0 (or 1 if 1-indexed), and Black's back rank (row 9, 'A') is rank 9 (or 10).
    """
    fr, fc = move.from_rc
    tr, tc = move.to_rc
    offset = 0 if zero_indexed_ranks else 1
    return f"{UCI_FILES[fc]}{fr + offset}{UCI_FILES[tc]}{tr + offset}"


def uci_to_move(board: XiangqiBoard, uci_str: str, zero_indexed_ranks: bool = True) -> Optional[Move]:
    """
    Parse a UCI/UCCI move string (e.g., 'h2e2' or 'b0c2') and return the matching
    legal `Move` on `board`, or `None` if no legal move matches.
    """
    s = uci_str.strip().lower()
    if len(s) < 4:
        return None
    fc = UCI_FILES.find(s[0])
    # Handle 1-or-2 digit ranks
    tc_idx = 2 if s[2] in UCI_FILES else 3
    tc = UCI_FILES.find(s[tc_idx])
    if fc == -1 or tc == -1:
        return None

    offset = 0 if zero_indexed_ranks else 1
    fr = int(s[1:tc_idx]) - offset
    tr = int(s[tc_idx + 1:]) - offset

    for move in board.get_legal_moves():
        if move.from_rc == (fr, fc) and move.to_rc == (tr, tc):
            return move
    return None


class ExternalUCIEngineAgent:
    """
    Adapter to benchmark against external UCI/UCCI Xiangqi engines (such as Fairy-Stockfish)
    via subprocess, implementing the same `select_move` protocol as `XiangqiAgent`.
    """

    def __init__(
        self,
        engine_path: str = "fairy-stockfish",
        movetime_ms: int = 500,
        zero_indexed_ranks: bool = False,
        name: str = "ExternalUCI",
    ) -> None:
        self.engine_path = engine_path
        self.movetime_ms = movetime_ms
        self.zero_indexed_ranks = zero_indexed_ranks
        self.name = name
        self.last_result: Optional[SearchResult] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._proc is None:
            self._proc = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._send("uci")
            self._send("setoption name UCI_Variant value xiangqi")
            self._send("isready")
            self._wait_for("readyok")

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._send("quit")
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

    def reset(self) -> None:
        self.last_result = None
        if self._proc is not None:
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok")

    def record_board(self, board: XiangqiBoard) -> None:
        pass

    def _send(self, cmd: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, token: str) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line or token in line:
                return line.strip()

    def select_move(self, board: XiangqiBoard) -> Optional[Move]:
        self.start()
        fen = board_to_fen(board)
        self._send(f"position fen {fen}")
        self._send(f"go movetime {self.movetime_ms}")
        bestmove_line = self._wait_for("bestmove")
        parts = bestmove_line.split()
        if len(parts) < 2 or parts[1] in ("(none)", "0000"):
            return None
        uci_move = parts[1]
        move = uci_to_move(board, uci_move, zero_indexed_ranks=self.zero_indexed_ranks)
        self.last_result = SearchResult(best_move=move, score=0, nodes_searched=0)
        return move