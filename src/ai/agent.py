import random
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.ai.board import BLACK, RED, Move, XiangqiBoard
from src.ai.eval import evaluate_board, evaluate_material
from src.ai.search import AlphaBetaEngine, SearchResult


class XiangqiAgent:
    """
    High-level AI agent wrapping AlphaBetaEngine with Iterative Deepening,
    configurable heuristics, and direct compatibility with game.py's BOARD_STATE.
    """

    def __init__(
        self,
        name: str = "XiangqiAI",
        max_depth: int = 3,
        time_limit: Optional[float] = 2.0,
        use_tt: bool = True,
        use_quiescence: bool = True,
        eval_fn: Callable[[XiangqiBoard, str], int] = evaluate_board,
    ) -> None:
        self.name = name
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.engine = AlphaBetaEngine(
            use_tt=use_tt,
            use_quiescence=use_quiescence,
            eval_fn=eval_fn,
        )
        self.game_history: Set[int] = set()
        self.last_result: Optional[SearchResult] = None

    def reset(self) -> None:
        """Clear search tables and position history for a new game."""
        self.engine.clear_tables()
        self.game_history.clear()
        self.last_result = None

    def record_board(self, board: XiangqiBoard) -> None:
        """Record the Zobrist hash of `board` in the agent's game history."""
        self.game_history.add(self.engine.hasher.compute_hash(board))

    def select_move(self, board: XiangqiBoard) -> Optional[Move]:
        """Select the best legal `Move` for `board.side_to_move`."""
        self.record_board(board)
        if self.time_limit is not None:
            result = self.engine.iterative_deepening_search(
                board,
                max_depth=self.max_depth,
                time_limit=self.time_limit,
                game_history=self.game_history,
            )
        else:
            result = self.engine.search(
                board,
                depth=self.max_depth,
                game_history=self.game_history,
            )
        self.last_result = result
        return result.best_move

    def select_gui_move(
        self,
        board_state: Dict[str, str],
        side_to_move: str = BLACK,
    ) -> Optional[Tuple[str, str]]:
        """
        Given a `BOARD_STATE` dictionary from `src/gui/game.py`, compute the best move
        for `side_to_move` and return `(piece_name, target_pos_str)` directly compatible
        with `move_piece(piece, pos)` in `game.py` (e.g., `('BC1', 'C5')`).
        """
        board = XiangqiBoard(board_state, side_to_move=side_to_move)
        move = self.select_move(board)
        if move is None:
            return None
        board.make_move(move)
        self.record_board(board)
        return move.piece, move.to_pos


class RandomAgent:
    """Baseline agent that chooses uniformly at random among legal moves."""

    def __init__(self, name: str = "RandomBot", seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.last_result: Optional[SearchResult] = None

    def reset(self) -> None:
        self.last_result = None

    def record_board(self, board: XiangqiBoard) -> None:
        pass

    def select_move(self, board: XiangqiBoard) -> Optional[Move]:
        legal = board.get_legal_moves()
        if not legal:
            return None
        chosen = self.rng.choice(legal)
        self.last_result = SearchResult(best_move=chosen, score=0, nodes_searched=len(legal), depth_reached=1)
        return chosen


def create_baseline_material_agent(max_depth: int = 2, time_limit: Optional[float] = None) -> XiangqiAgent:
    """Factory for a baseline Material-Only Alpha-Beta agent (no TT, no Quiescence)."""
    return XiangqiAgent(
        name=f"BaselineMaterial(d={max_depth})",
        max_depth=max_depth,
        time_limit=time_limit,
        use_tt=False,
        use_quiescence=False,
        eval_fn=evaluate_material,
    )


def get_legal_gui_targets(board_state: Dict[str, str], piece: str) -> List[str]:
    """
    Helper for `game.py`: return all strictly legal destination strings (e.g., `['H5', 'E2']`)
    for `piece` in the given `board_state`.
    """
    pos = board_state.get(piece, "")
    if not pos:
        return []
    color = piece[0]
    board = XiangqiBoard(board_state, side_to_move=color)
    return [m.to_pos for m in board.get_legal_moves(color) if m.piece == piece]


def is_valid_gui_move(board_state: Dict[str, str], piece: str, target_pos: str) -> bool:
    """Return True if moving `piece` to `target_pos` is a strictly legal Xiangqi move."""
    return target_pos in get_legal_gui_targets(board_state, piece)