import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.ai.board import Move, XiangqiBoard
from src.ai.eval import PIECE_VALUES, evaluate_board
from src.ai.transposition import TranspositionTable, TTFlag, ZobristHasher

MATE_SCORE = 100000
MATE_THRESHOLD = 99000
INF = 1000000
MAX_PLY = 128

# Move ordering priority offsets
TT_MOVE_BONUS = 10_000_000
CAPTURE_BASE_BONUS = 1_000_000
KILLER_1_BONUS = 900_000
KILLER_2_BONUS = 800_000


@dataclass
class SearchResult:
    best_move: Optional[Move]
    score: int
    nodes_searched: int
    depth_reached: int = 0
    tt_hits: int = 0
    q_nodes_searched: int = 0
    elapsed_time: float = 0.0


def mvv_lva_score(move: Move) -> int:
    """
    Score a move using Most Valuable Victim - Least Valuable Attacker (MVV-LVA).
    Prioritizes capturing high-value opponent pieces with low-value pieces.
    Quiet moves receive a score of 0.
    """
    if move.captured is None:
        return 0
    victim_val = PIECE_VALUES[move.captured[1]]
    attacker_val = PIECE_VALUES[move.piece[1]]
    return 10 * victim_val - attacker_val


def order_moves(
    moves: List[Move],
    tt_move: Optional[Move] = None,
    killers: Optional[List[Optional[Move]]] = None,
    history_table: Optional[Dict[Tuple[str, Tuple[int, int], Tuple[int, int]], int]] = None,
) -> List[Move]:
    """
    Rank moves using:
    1. Transposition Table best move
    2. Captures ordered by MVV-LVA
    3. Killer moves (quiet moves that caused a beta cutoff at the same ply)
    4. History heuristic score (quiet moves that succeeded across the tree)
    """
    def move_priority(move: Move) -> int:
        if tt_move is not None and move == tt_move:
            return TT_MOVE_BONUS
        if move.captured is not None:
            return CAPTURE_BASE_BONUS + mvv_lva_score(move)
        if killers is not None:
            if killers[0] is not None and move == killers[0]:
                return KILLER_1_BONUS
            if len(killers) > 1 and killers[1] is not None and move == killers[1]:
                return KILLER_2_BONUS
        if history_table is not None:
            return min(history_table.get((move.piece, move.from_rc, move.to_rc), 0), KILLER_2_BONUS - 1)
        return 0

    return sorted(moves, key=move_priority, reverse=True)


def _score_to_tt(score: int, ply: int) -> int:
    """Convert a ply-relative mate score to a node-relative mate score for TT storage."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    """Convert a stored TT mate score back to a root-ply-relative mate score."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


class AlphaBetaEngine:
    """
    Minimax engine with Negamax Alpha-Beta pruning, Iterative Deepening,
    Zobrist Transposition Table, MVV-LVA, Killer/History Heuristics, and Quiescence Search.
    """

    def __init__(
        self,
        tt_size: int = 200_000,
        use_tt: bool = True,
        use_quiescence: bool = True,
        max_q_depth: int = 6,
        eval_fn: Callable[[XiangqiBoard, str], int] = evaluate_board,
    ) -> None:
        self.nodes_searched: int = 0
        self.q_nodes_searched: int = 0
        self.use_tt: bool = use_tt
        self.use_quiescence: bool = use_quiescence
        self.max_q_depth: int = max_q_depth
        self.eval_fn: Callable[[XiangqiBoard, str], int] = eval_fn
        self.hasher = ZobristHasher()
        self.tt = TranspositionTable(max_entries=tt_size)
        self.killer_moves: List[List[Optional[Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.history_table: Dict[Tuple[str, Tuple[int, int], Tuple[int, int]], int] = {}
        self._stop_time: Optional[float] = None
        self._aborted: bool = False
        self._rep_stack: Set[int] = set()

    def clear_tables(self) -> None:
        """Reset Transposition, Killer, and History tables."""
        self.tt.clear()
        self.killer_moves = [[None, None] for _ in range(MAX_PLY)]
        self.history_table.clear()

    def search(
        self,
        board: XiangqiBoard,
        depth: int,
        game_history: Optional[Set[int]] = None,
    ) -> SearchResult:
        """Fixed-depth Alpha-Beta search for `board.side_to_move`."""
        start_t = time.perf_counter()
        self.nodes_searched = 0
        self.q_nodes_searched = 0
        self.tt.hits = 0
        self._stop_time = None
        self._aborted = False
        self._rep_stack = set(game_history) if game_history else set()
        best_move, best_score = self._alphabeta_root(board, depth)
        return SearchResult(
            best_move=best_move,
            score=best_score,
            nodes_searched=self.nodes_searched,
            depth_reached=depth,
            tt_hits=self.tt.hits,
            q_nodes_searched=self.q_nodes_searched,
            elapsed_time=time.perf_counter() - start_t,
        )

    def iterative_deepening_search(
        self,
        board: XiangqiBoard,
        max_depth: int,
        time_limit: Optional[float] = None,
        game_history: Optional[Set[int]] = None,
    ) -> SearchResult:
        """
        Run Iterative Deepening from depth 1 up to `max_depth`.
        If `time_limit` (in seconds) is provided, stops once the budget expires
        and returns the result from the deepest completed iteration.
        """
        start_t = time.perf_counter()
        self.nodes_searched = 0
        self.q_nodes_searched = 0
        self.tt.hits = 0
        self._aborted = False
        self._stop_time = (start_t + time_limit) if time_limit is not None else None
        self._rep_stack = set(game_history) if game_history else set()

        best_result = SearchResult(
            best_move=None,
            score=0,
            nodes_searched=0,
            depth_reached=0,
            tt_hits=0,
            q_nodes_searched=0,
            elapsed_time=0.0,
        )

        for current_depth in range(1, max_depth + 1):
            move, score = self._alphabeta_root(board, current_depth)
            if self._aborted and current_depth > 1:
                break
            best_result = SearchResult(
                best_move=move,
                score=score,
                nodes_searched=self.nodes_searched,
                depth_reached=current_depth,
                tt_hits=self.tt.hits,
                q_nodes_searched=self.q_nodes_searched,
                elapsed_time=time.perf_counter() - start_t,
            )
            if abs(score) >= MATE_THRESHOLD:
                break
            if self._stop_time is not None and time.perf_counter() >= self._stop_time:
                break

        best_result.elapsed_time = time.perf_counter() - start_t
        return best_result

    def _record_quiet_cutoff(self, move: Move, depth: int, ply: int) -> None:
        """Update Killer and History tables when a quiet move causes a beta cutoff."""
        if move.captured is not None or ply >= MAX_PLY:
            return
        if self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
        key = (move.piece, move.from_rc, move.to_rc)
        self.history_table[key] = self.history_table.get(key, 0) + depth * depth

    def _alphabeta_root(self, board: XiangqiBoard, depth: int) -> Tuple[Optional[Move], int]:
        zobrist_key = self.hasher.compute_hash(board)
        tt_entry = self.tt.probe(zobrist_key) if self.use_tt else None
        if tt_entry is not None:
            self.tt.hits += 1
        tt_move = tt_entry.best_move if tt_entry is not None else None

        legal_moves = order_moves(
            board.get_legal_moves(),
            tt_move=tt_move,
            killers=self.killer_moves[0],
            history_table=self.history_table,
        )
        if not legal_moves:
            return None, -MATE_SCORE

        best_move: Optional[Move] = legal_moves[0]
        best_score = -INF
        alpha = -INF
        beta = INF

        added_root = zobrist_key not in self._rep_stack
        self._rep_stack.add(zobrist_key)
        try:
            for move in legal_moves:
                board.make_move(move)
                child_hash = self.hasher.compute_hash(board)
                if child_hash in self._rep_stack:
                    # Treat repeating an earlier position as a draw (0) to discourage loops when winning
                    score = 0
                else:
                    score = -self._alphabeta(board, depth - 1, -beta, -alpha, ply=1)
                board.undo_move()

                if self._aborted:
                    return best_move, best_score

                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
        finally:
            if added_root:
                self._rep_stack.discard(zobrist_key)

        if self.use_tt and not self._aborted:
            self.tt.store(
                key=zobrist_key,
                depth=depth,
                score=_score_to_tt(best_score, ply=0),
                flag=TTFlag.EXACT,
                best_move=best_move,
            )

        return best_move, best_score

    def _alphabeta(self, board: XiangqiBoard, depth: int, alpha: int, beta: int, ply: int) -> int:
        if self._stop_time is not None and (self.nodes_searched & 255) == 0:
            if time.perf_counter() >= self._stop_time:
                self._aborted = True
                return 0

        self.nodes_searched += 1
        alpha_orig = alpha

        zobrist_key = self.hasher.compute_hash(board)
        if ply > 0 and zobrist_key in self._rep_stack:
            return 0

        tt_move: Optional[Move] = None

        if self.use_tt:
            tt_entry = self.tt.probe(zobrist_key)
            if tt_entry is not None:
                self.tt.hits += 1
                tt_move = tt_entry.best_move
                if tt_entry.depth >= depth:
                    tt_score = _score_from_tt(tt_entry.score, ply)
                    if tt_entry.flag == TTFlag.EXACT:
                        return tt_score
                    elif tt_entry.flag == TTFlag.LOWERBOUND:
                        alpha = max(alpha, tt_score)
                    elif tt_entry.flag == TTFlag.UPPERBOUND:
                        beta = min(beta, tt_score)
                    if alpha >= beta:
                        return tt_score

        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return -MATE_SCORE + ply

        if depth <= 0:
            if self.use_quiescence:
                leaf_score = self._quiescence(board, alpha, beta, ply, q_depth=0, legal_moves=legal_moves)
            else:
                leaf_score = self.eval_fn(board, board.side_to_move)

            if self.use_tt and not self._aborted:
                if leaf_score <= alpha_orig:
                    flag = TTFlag.UPPERBOUND
                elif leaf_score >= beta:
                    flag = TTFlag.LOWERBOUND
                else:
                    flag = TTFlag.EXACT
                self.tt.store(
                    key=zobrist_key,
                    depth=0,
                    score=_score_to_tt(leaf_score, ply),
                    flag=flag,
                    best_move=None,
                )
            return leaf_score

        ordered = order_moves(
            legal_moves,
            tt_move=tt_move,
            killers=self.killer_moves[ply] if ply < MAX_PLY else None,
            history_table=self.history_table,
        )

        best_score = -INF
        best_move: Optional[Move] = ordered[0]

        added_node = zobrist_key not in self._rep_stack
        self._rep_stack.add(zobrist_key)
        try:
            for move in ordered:
                board.make_move(move)
                score = -self._alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
                board.undo_move()

                if self._aborted:
                    return 0

                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self._record_quiet_cutoff(move, depth, ply)
                    break  # Beta cutoff
        finally:
            if added_node:
                self._rep_stack.discard(zobrist_key)

        if self.use_tt and not self._aborted:
            if best_score <= alpha_orig:
                flag = TTFlag.UPPERBOUND
            elif best_score >= beta:
                flag = TTFlag.LOWERBOUND
            else:
                flag = TTFlag.EXACT
            self.tt.store(
                key=zobrist_key,
                depth=depth,
                score=_score_to_tt(best_score, ply),
                flag=flag,
                best_move=best_move,
            )

        return best_score

    def _quiescence(
        self,
        board: XiangqiBoard,
        alpha: int,
        beta: int,
        ply: int,
        q_depth: int,
        legal_moves: Optional[List[Move]] = None,
    ) -> int:
        """
        Quiescence search through tactical capture sequences to stabilize leaf nodes
        and mitigate the horizon effect.
        """
        if self._stop_time is not None and (self.nodes_searched & 255) == 0:
            if time.perf_counter() >= self._stop_time:
                self._aborted = True
                return 0

        if q_depth > 0:
            self.nodes_searched += 1
            self.q_nodes_searched += 1

        if legal_moves is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                return -MATE_SCORE + ply

        stand_pat = self.eval_fn(board, board.side_to_move)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        if q_depth >= self.max_q_depth:
            return stand_pat

        captures = [m for m in legal_moves if m.captured is not None]
        if not captures:
            return stand_pat

        best_score = stand_pat
        for move in order_moves(captures):
            board.make_move(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1, q_depth + 1)
            board.undo_move()

            if self._aborted:
                return 0

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score