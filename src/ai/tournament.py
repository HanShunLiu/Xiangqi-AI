import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from src.ai.agent import RandomAgent, XiangqiAgent, create_baseline_material_agent
from src.ai.board import BLACK, INITIAL_BOARD_STATE, RED, Move, XiangqiBoard
from src.ai.eval import EvalWeights, make_weighted_evaluator
from src.ai.search import SearchResult


class SupportsSelectMove(Protocol):
    name: str
    last_result: Optional[SearchResult]

    def reset(self) -> None: ...
    def record_board(self, board: XiangqiBoard) -> None: ...
    def select_move(self, board: XiangqiBoard) -> Optional[Move]: ...


# Diverse starting positions for multi-game tournaments so deterministic agents
# explore varied tactical lines across games.
OPENING_BOOK_STATES: List[Dict[str, str]] = [
    # 0. Standard initial board
    dict(INITIAL_BOARD_STATE),
    # 1. Central Cannon vs. Screen Horses (1. RC2 H8->H5, BH2 A8->C7)
    {**INITIAL_BOARD_STATE, "RC2": "H5", "BH2": "C7"},
    # 2. Left Central Cannon vs. Left Horse (1. RC1 H2->H5, BH1 A2->C3)
    {**INITIAL_BOARD_STATE, "RC1": "H5", "BH1": "C3"},
    # 3. Immortal's Guide (Pawn Opening: 1. RS2 G3->F3, BC2 C8->C5)
    {**INITIAL_BOARD_STATE, "RS2": "F3", "BC2": "C5"},
    # 4. Flying Elephant Opening (1. RE1 J3->H5, BC1 C2->C5)
    {**INITIAL_BOARD_STATE, "RE1": "H5", "BC1": "C5"},
]


@dataclass
class MatchSummary:
    red_name: str
    black_name: str
    winner: Optional[str]  # "R", "B", or None (draw by move cap / repetition)
    termination_reason: str
    total_plies: int
    red_nodes: int = 0
    black_nodes: int = 0
    red_tt_hits: int = 0
    black_tt_hits: int = 0
    red_time_sec: float = 0.0
    black_time_sec: float = 0.0
    move_log: List[str] = field(default_factory=list)


@dataclass
class SeriesSummary:
    agent_a_name: str
    agent_b_name: str
    num_games: int
    a_wins: int
    b_wins: int
    draws: int
    a_win_rate: float
    elo_diff_a_vs_b: float
    matches: List[MatchSummary] = field(default_factory=list)


@dataclass
class PruningBenchmarkReport:
    depth: int
    baseline_nodes: int
    optimized_nodes: int
    tt_hits: int
    q_nodes: int
    node_reduction_pct: float
    baseline_time_sec: float
    optimized_time_sec: float


def compute_elo_diff(wins: int, losses: int, draws: int) -> float:
    """
    Compute the estimated Elo rating difference (Player A - Player B)
    from match outcomes using the standard logistic Elo formula.
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0
    score_ratio = (wins + 0.5 * draws) / total
    # Clamp score_ratio to [0.001, 0.999] to avoid infinite log values on 100% sweeps
    clamped = min(0.999, max(0.001, score_ratio))
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def play_match(
    red_agent: SupportsSelectMove,
    black_agent: SupportsSelectMove,
    initial_state: Optional[Dict[str, str]] = None,
    max_plies: int = 200,
) -> MatchSummary:
    """
    Play an automated game between `red_agent` and `black_agent`.
    Tracks nodes searched, TT hits, and turn time for both sides.
    """
    board = XiangqiBoard(initial_state or INITIAL_BOARD_STATE, side_to_move=RED)
    red_agent.reset()
    black_agent.reset()

    summary = MatchSummary(
        red_name=red_agent.name,
        black_name=black_agent.name,
        winner=None,
        termination_reason="max_plies",
        total_plies=0,
    )

    state_counts: Dict[int, int] = {}

    for ply in range(max_plies):
        if hasattr(red_agent, "engine"):
            z_hash = red_agent.engine.hasher.compute_hash(board)
        elif hasattr(black_agent, "engine"):
            z_hash = black_agent.engine.hasher.compute_hash(board)
        else:
            z_hash = hash(str(board.to_board_state()) + board.side_to_move)

        state_counts[z_hash] = state_counts.get(z_hash, 0) + 1
        if state_counts[z_hash] >= 3:
            summary.termination_reason = "threefold_repetition"
            summary.total_plies = ply
            return summary

        is_over, winner = board.is_game_over()
        if is_over:
            summary.winner = winner
            summary.termination_reason = "checkmate_or_stalemate"
            summary.total_plies = ply
            return summary

        current_agent = red_agent if board.side_to_move == RED else black_agent
        t0 = time.perf_counter()
        move = current_agent.select_move(board)
        elapsed = time.perf_counter() - t0

        if move is None:
            summary.winner = board.opponent(board.side_to_move)
            summary.termination_reason = "no_legal_moves"
            summary.total_plies = ply
            return summary

        res = current_agent.last_result
        if board.side_to_move == RED:
            summary.red_time_sec += elapsed
            if res is not None:
                summary.red_nodes += res.nodes_searched
                summary.red_tt_hits += res.tt_hits
        else:
            summary.black_time_sec += elapsed
            if res is not None:
                summary.black_nodes += res.nodes_searched
                summary.black_tt_hits += res.tt_hits

        board.make_move(move)
        red_agent.record_board(board)
        black_agent.record_board(board)
        summary.move_log.append(repr(move))
        summary.total_plies = ply + 1

    return summary


def run_tournament_series(
    agent_a: SupportsSelectMove,
    agent_b: SupportsSelectMove,
    num_games: int = 4,
    max_plies: int = 120,
    opening_states: Optional[List[Dict[str, str]]] = None,
) -> SeriesSummary:
    """
    Run a multi-game tournament between `agent_a` and `agent_b`, alternating
    Red/Black colors across games and cycling through `opening_states`.
    Returns aggregate win/loss/draw counts, win rate, and Elo difference.
    """
    openings = opening_states or OPENING_BOOK_STATES
    a_wins = 0
    b_wins = 0
    draws = 0
    matches: List[MatchSummary] = []

    for i in range(num_games):
        state = openings[(i // 2) % len(openings)]
        a_is_red = (i % 2 == 0)
        red_bot = agent_a if a_is_red else agent_b
        black_bot = agent_b if a_is_red else agent_a

        m = play_match(red_bot, black_bot, initial_state=state, max_plies=max_plies)
        matches.append(m)

        if m.winner is None:
            draws += 1
        elif (m.winner == RED and a_is_red) or (m.winner == BLACK and not a_is_red):
            a_wins += 1
        else:
            b_wins += 1

    win_rate = (a_wins + 0.5 * draws) / max(1, num_games)
    elo_diff = compute_elo_diff(a_wins, b_wins, draws)

    return SeriesSummary(
        agent_a_name=agent_a.name,
        agent_b_name=agent_b.name,
        num_games=num_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_win_rate=win_rate,
        elo_diff_a_vs_b=elo_diff,
        matches=matches,
    )


def benchmark_pruning_efficiency(
    board_state: Optional[Dict[str, str]] = None,
    depth: int = 3,
) -> PruningBenchmarkReport:
    """
    Compare search efficiency at `depth` between:
    1. Baseline Alpha-Beta (no Transposition Table, no Iterative Deepening, no Quiescence)
    2. Optimized Alpha-Beta (Iterative Deepening + Zobrist Transposition Table)
    """
    board1 = XiangqiBoard(board_state or INITIAL_BOARD_STATE, side_to_move=RED)
    board2 = XiangqiBoard(board_state or INITIAL_BOARD_STATE, side_to_move=RED)

    baseline = XiangqiAgent(
        name="Baseline",
        max_depth=depth,
        time_limit=None,
        use_tt=False,
        use_quiescence=False,
    )
    optimized = XiangqiAgent(
        name="Optimized",
        max_depth=depth,
        time_limit=60.0,
        use_tt=True,
        use_quiescence=False,
    )

    baseline.select_move(board1)
    optimized.select_move(board2)

    b_res = baseline.last_result
    o_res = optimized.last_result
    assert b_res is not None and o_res is not None

    reduction = 100.0 * (1.0 - (o_res.nodes_searched / max(1, b_res.nodes_searched)))
    return PruningBenchmarkReport(
        depth=depth,
        baseline_nodes=b_res.nodes_searched,
        optimized_nodes=o_res.nodes_searched,
        tt_hits=o_res.tt_hits,
        q_nodes=o_res.q_nodes_searched,
        node_reduction_pct=reduction,
        baseline_time_sec=b_res.elapsed_time,
        optimized_time_sec=o_res.elapsed_time,
    )


def run_ablation_study(depth: int = 2, num_games: int = 2, max_plies: int = 80) -> List[SeriesSummary]:
    """
    Compare the full Heuristic AI against ablated variants:
    1. Full Heuristic AI vs. Baseline Material-Only AI
    2. Full Heuristic AI vs. No-Positional-PST AI
    """
    full_ai = XiangqiAgent(name=f"FullHeuristic(d={depth})", max_depth=depth, time_limit=2.0)
    material_ai = create_baseline_material_agent(max_depth=depth)
    no_pst_ai = XiangqiAgent(
        name=f"NoPST_NoSafety(d={depth})",
        max_depth=depth,
        time_limit=2.0,
        eval_fn=make_weighted_evaluator(EvalWeights(pst=0.0, king_safety=0.0)),
    )

    results = [
        run_tournament_series(full_ai, material_ai, num_games=num_games, max_plies=max_plies),
        run_tournament_series(full_ai, no_pst_ai, num_games=num_games, max_plies=max_plies),
    ]
    return results


if __name__ == "__main__":
    print("=== Running Pruning Efficiency Benchmark (Depth 3) ===")
    report = benchmark_pruning_efficiency(depth=3)
    print(
        f"Baseline nodes: {report.baseline_nodes} ({report.baseline_time_sec:.3f}s) | "
        f"Optimized ID+TT nodes: {report.optimized_nodes} ({report.optimized_time_sec:.3f}s) | "
        f"TT hits: {report.tt_hits}"
    )

    print("\n=== Running Sample Series: Full Heuristic AI vs. RandomBot ===")
    ai_bot = XiangqiAgent(name="HeuristicAI(d=2)", max_depth=2, time_limit=1.0)
    rand_bot = RandomAgent(name="RandomBot", seed=42)
    series = run_tournament_series(ai_bot, rand_bot, num_games=2, max_plies=60)
    print(
        f"{series.agent_a_name} vs {series.agent_b_name}: "
        f"+{series.a_wins} -{series.b_wins} ={series.draws} "
        f"(Score: {series.a_win_rate * 100:.1f}%, Est. Elo Diff: {series.elo_diff_a_vs_b:+.0f})"
    )