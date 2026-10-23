import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

from src.ai.board import BLACK, RED, Move, XiangqiBoard

PIECE_TYPES = ("G", "A", "E", "H", "R", "C", "S")
COLORS = (RED, BLACK)


class TTFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1  # Beta cutoff (score >= beta)
    UPPERBOUND = 2  # Failed low (score <= alpha)


@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[Move]


class ZobristHasher:
    """Deterministic 64-bit Zobrist hasher for Xiangqi board states."""

    def __init__(self, seed: int = 20260401) -> None:
        rng = random.Random(seed)
        self.piece_keys: Dict[Tuple[str, str, int, int], int] = {}
        for color in COLORS:
            for ptype in PIECE_TYPES:
                for r in range(10):
                    for c in range(9):
                        self.piece_keys[(color, ptype, r, c)] = rng.getrandbits(64)
        self.black_to_move_key: int = rng.getrandbits(64)

    def compute_hash(self, board: XiangqiBoard) -> int:
        """Compute the 64-bit Zobrist hash of `board`."""
        h = 0
        for piece, rc in board.piece_coords.items():
            if rc is not None:
                r, c = rc
                h ^= self.piece_keys[(piece[0], piece[1], r, c)]
        if board.side_to_move == BLACK:
            h ^= self.black_to_move_key
        return h


class TranspositionTable:
    """Fixed-capacity Transposition Table with depth-preferred replacement."""

    def __init__(self, max_entries: int = 200_000) -> None:
        self.max_entries = max_entries
        self.table: Dict[int, TTEntry] = {}
        self.hits: int = 0

    def clear(self) -> None:
        self.table.clear()
        self.hits = 0

    def probe(self, key: int) -> Optional[TTEntry]:
        entry = self.table.get(key)
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: TTFlag,
        best_move: Optional[Move],
    ) -> None:
        existing = self.table.get(key)
        # Replace if not present, or if new search is at least as deep as existing entry
        if existing is None or depth >= existing.depth:
            if existing is None and len(self.table) >= self.max_entries:
                # Simple eviction of oldest inserted key when table reaches capacity
                oldest_key = next(iter(self.table))
                del self.table[oldest_key]
            self.table[key] = TTEntry(
                key=key,
                depth=depth,
                score=score,
                flag=flag,
                best_move=best_move,
            )