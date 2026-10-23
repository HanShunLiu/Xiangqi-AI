import unittest
from src.ai import (
    BLACK,
    INITIAL_BOARD_STATE,
    RED,
    RandomAgent,
    XiangqiAgent,
    create_baseline_material_agent,
    get_legal_gui_targets,
    is_valid_gui_move,
)
from src.ai.tournament import benchmark_pruning_efficiency, play_match


class TestAgentAndTournament(unittest.TestCase):
    def test_gui_legal_targets_and_validation(self):
        """Verify GUI bridge helpers work directly on game.py's BOARD_STATE dict."""
        targets = get_legal_gui_targets(INITIAL_BOARD_STATE, "RH1")
        # Red Horse RH1 at J2 can jump to H1 or H3 in the initial setup
        self.assertEqual(set(targets), {"H1", "H3"})
        self.assertTrue(is_valid_gui_move(INITIAL_BOARD_STATE, "RH1", "H3"))
        self.assertFalse(is_valid_gui_move(INITIAL_BOARD_STATE, "RH1", "I4"))  # Blocked by RE1 at J3

    def test_select_gui_move_returns_valid_piece_and_pos(self):
        """select_gui_move should return a (piece, pos) tuple ready for game.py's move_piece()."""
        agent = XiangqiAgent(max_depth=2, time_limit=1.0)
        move_tuple = agent.select_gui_move(INITIAL_BOARD_STATE, side_to_move=BLACK)
        self.assertIsNotNone(move_tuple)
        piece, target_pos = move_tuple
        self.assertTrue(piece.startswith("B"))
        self.assertTrue(is_valid_gui_move(INITIAL_BOARD_STATE, piece, target_pos))

    def test_bot_vs_bot_endgame_match_completion(self):
        """Verify play_match runs a bot-vs-bot game to checkmate and logs telemetry."""
        endgame_state = {
            "RG_": "J4",
            "RR1": "B2",
            "RR2": "C2",
            "BG_": "A5",
            "BS1": "D1",
        }
        red_ai = XiangqiAgent(name="RedHeuristic", max_depth=3, time_limit=2.0)
        black_ai = create_baseline_material_agent(max_depth=2)

        summary = play_match(red_ai, black_ai, initial_state=endgame_state, max_plies=10)
        self.assertEqual(summary.winner, RED)
        self.assertEqual(summary.termination_reason, "checkmate_or_stalemate")
        self.assertLessEqual(summary.total_plies, 3)
        self.assertGreater(summary.red_nodes, 0)

    def test_heuristic_agent_defeats_random_agent(self):
        """Full heuristic AI should quickly defeat RandomAgent from a tactical middlegame state."""
        tactical_state = {
            "RG_": "J5",
            "RA1": "J4",
            "RA2": "J6",
            "RR1": "C1",
            "RR2": "C9",
            "RC1": "H5",
            "BG_": "A5",
            "BS1": "D1",
            "BS2": "D9",
        }
        red_ai = XiangqiAgent(name="HeuristicAI", max_depth=2, time_limit=1.0)
        black_rand = RandomAgent(name="RandomBot", seed=7)

        summary = play_match(red_ai, black_rand, initial_state=tactical_state, max_plies=20)
        self.assertEqual(summary.winner, RED)

    def test_pruning_benchmark_report(self):
        """Ensure benchmark_pruning_efficiency runs and records TT hits."""
        custom_state = {
            "RG_": "J5",
            "RA1": "J4",
            "RR1": "E4",
            "RC1": "H5",
            "BG_": "A5",
            "BA1": "A4",
            "BR1": "D6",
            "BC1": "C5",
            "BS1": "D5",
        }
        report = benchmark_pruning_efficiency(board_state=custom_state, depth=3)
        self.assertEqual(report.depth, 3)
        self.assertGreater(report.baseline_nodes, 0)
        self.assertGreater(report.optimized_nodes, 0)
        self.assertGreater(report.tt_hits, 0)

    def test_pst_and_eval_weights_ablation(self):
        """Verify Piece-Square Tables and EvalWeights can toggle heuristic components."""
        from src.ai.eval import EvalWeights, get_pst_bonus, make_weighted_evaluator
        from src.ai.board import XiangqiBoard, pos_to_coords

        # Central advanced soldier at D5 should have a higher PST score than a flank soldier at G1
        r_adv, c_adv = pos_to_coords("D5")
        r_base, c_base = pos_to_coords("G1")
        self.assertGreater(get_pst_bonus("RS3", r_adv, c_adv), get_pst_bonus("RS1", r_base, c_base))

        # Verify custom weighted evaluator runs cleanly in XiangqiAgent
        weights_no_king_safety = EvalWeights(king_safety=0.0, palace_infiltration=2.0)
        custom_eval = make_weighted_evaluator(weights_no_king_safety)
        agent = XiangqiAgent(name="AblationAgent", max_depth=2, time_limit=None, eval_fn=custom_eval)
        board = XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)
        move = agent.select_move(board)
        self.assertIsNotNone(move)

    def test_fen_roundtrip_and_uci_conversion(self):
        """Verify converting XiangqiBoard -> FEN -> XiangqiBoard and Move <-> UCI notation."""
        from src.ai.fen_uci import board_to_fen, fen_to_board, move_to_uci, uci_to_move
        from src.ai.board import XiangqiBoard

        board = XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)
        fen = board_to_fen(board)
        self.assertTrue(fen.startswith("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"))

        reconstructed = fen_to_board(fen)
        self.assertEqual(reconstructed.side_to_move, RED)
        self.assertEqual( len(reconstructed.get_legal_moves(RED)), 44)

        # Test UCI move roundtrip on opening cannon move RC1: H2 -> H5
        sample_move = next(m for m in board.get_legal_moves(RED) if m.piece == "RC1" and m.to_pos == "H5")
        uci_0 = move_to_uci(sample_move, zero_indexed_ranks=True)
        parsed_0 = uci_to_move(board, uci_0, zero_indexed_ranks=True)
        self.assertEqual(parsed_0, sample_move)

        uci_1 = move_to_uci(sample_move, zero_indexed_ranks=False)
        parsed_1 = uci_to_move(board, uci_1, zero_indexed_ranks=False)
        self.assertEqual(parsed_1, sample_move)

    def test_tournament_series_and_elo_calculation(self):
        """Verify run_tournament_series alternates colors and computes Elo differences."""
        from src.ai.tournament import compute_elo_diff, run_tournament_series

        self.assertAlmostEqual(compute_elo_diff(5, 5, 0), 0.0, places=2)
        self.assertGreater(compute_elo_diff(8, 2, 0), 200.0)

        tactical_openings = [
            {
                "RG_": "J5", "RA1": "J4", "RR1": "C1", "RR2": "C9",
                "BG_": "A5", "BA1": "A4", "BR1": "H1", "BR2": "H9",
            }
        ]
        bot_a = XiangqiAgent(name="HeuristicD2", max_depth=2, time_limit=1.0)
        bot_b = RandomAgent(name="RandomBot", seed=123)

        series = run_tournament_series(
            bot_a,
            bot_b,
            num_games=2,
            max_plies=30,
            opening_states=tactical_openings,
        )
        self.assertEqual(series.num_games, 2)
        self.assertEqual(len(series.matches), 2)
        self.assertGreaterEqual(series.a_wins, 1)
        self.assertGreater(series.elo_diff_a_vs_b, 0.0)

if __name__ == "__main__":
    unittest.main()