import unittest
from src.ai.board import BLACK, INITIAL_BOARD_STATE, RED, Move, XiangqiBoard, pos_to_coords
from src.ai.eval import evaluate_board, evaluate_material
from src.ai.search import MATE_SCORE, AlphaBetaEngine, order_moves


class TestEvalAndSearch(unittest.TestCase):
    def test_initial_board_evaluation_is_zero(self):
        """Symmetric starting board should evaluate to 0 for both sides."""
        board = XiangqiBoard(INITIAL_BOARD_STATE)
        self.assertEqual(evaluate_board(board, RED), 0)
        self.assertEqual(evaluate_board(board, BLACK), 0)

    def test_material_advantage_and_river_bonus(self):
        """Red Soldier across the river should grant base value (100) + river bonus (100)."""
        state = {
            "RG_": "J5",
            "BG_": "A4",
            "RS1": "E5",  # Red Soldier across the river (worth 200)
            "BS1": "D1",  # Black Soldier on own side (worth 100)
        }
        board = XiangqiBoard(state, side_to_move=RED)
        self.assertEqual(evaluate_material(board, RED), 100)
        self.assertEqual(evaluate_material(board, BLACK), -100)

    def test_dynamic_material_and_positional_heuristics(self):
        """
        Verify positional heuristics:
        1. Palace infiltration scores higher than passive back-rank placement.
        2. Intact Advisor/Elephant guard improves King safety.
        """
        active_red = {
            "RG_": "J5",
            "RA1": "J4",
            "RA2": "J6",
            "RE1": "J3",
            "RE2": "J7",
            "RR1": "B4",  # Infiltrated Black's palace!
            "BG_": "A5",
            "BR1": "A1",  # Passive corner chariot, no guards
        }
        passive_red = {
            "RG_": "J5",
            "RR1": "J1",  # Passive corner chariot
            "BG_": "A5",
            "BR1": "A1",
        }
        board_active = XiangqiBoard(active_red, side_to_move=RED)
        board_passive = XiangqiBoard(passive_red, side_to_move=RED)
        self.assertGreater(evaluate_board(board_active, RED), evaluate_board(board_passive, RED))

    def test_mvv_lva_move_ordering(self):
        """Soldier capturing Chariot should be ordered before Chariot capturing Soldier."""
        low_takes_high = Move("RS1", pos_to_coords("E5"), pos_to_coords("D5"), captured="BR1")
        high_takes_low = Move("RR1", pos_to_coords("J1"), pos_to_coords("D1"), captured="BS1")
        quiet_move = Move("RA1", pos_to_coords("J4"), pos_to_coords("I5"), captured=None)

        ordered = order_moves([quiet_move, high_takes_low, low_takes_high])
        self.assertEqual(ordered[0], low_takes_high)
        self.assertEqual(ordered[1], high_takes_low)
        self.assertEqual(ordered[2], quiet_move)

    def test_captures_hanging_chariot(self):
        """Engine should immediately capture an undefended Black Chariot."""
        state = {
            "RG_": "J5",
            "BG_": "A4",
            "RR1": "E1",
            "BR1": "E9",  # Hanging on rank E
        }
        board = XiangqiBoard(state, side_to_move=RED)
        engine = AlphaBetaEngine()
        result = engine.search(board, depth=2)
        self.assertIsNotNone(result.best_move)
        self.assertEqual(result.best_move.piece, "RR1")
        self.assertEqual(result.best_move.to_pos, "E9")
        self.assertEqual(result.best_move.captured, "BR1")

    def test_quiescence_prevents_horizon_effect_blunder(self):
        """
        At depth=1 without Quiescence, Red Chariot (RR1 at E2) might capture a defended
        Black Soldier (BS1 at E5, protected by Black Chariot BR1 at E9), thinking it won
        a pawn because the recapture happens at ply 2.
        With Quiescence enabled, the engine sees the recapture RR1xBS1 -> BR1xRR1 at depth=1
        and avoids blundering its Chariot.
        """
        state = {
            "RG_": "J5",
            "BG_": "A4",
            "RR1": "E2",
            "BS1": "E5",  # Defended by BR1 on E9
            "BR1": "E9",
        }
        board = XiangqiBoard(state, side_to_move=RED)
        engine_q = AlphaBetaEngine(use_quiescence=True)
        res_q = engine_q.search(board, depth=1)

        self.assertIsNotNone(res_q.best_move)
        # Must NOT capture the defended soldier at E5 with the Chariot
        self.assertNotEqual((res_q.best_move.piece, res_q.best_move.to_pos), ("RR1", "E5"))
        self.assertGreater(res_q.q_nodes_searched, 0)

    def test_mate_in_one_detection(self):
        """Engine should find the unique 1-move checkmate and prefer it over slower mates."""
        state = {
            "RG_": "J4",
            "BG_": "A5",
            "BS1": "D1",  # Prevents 1-move non-checking stalemate so Red must deliver checkmate
            "RR1": "B1",  # Locks BG_ onto rank A
            "RR2": "C9",  # Moving RR2 to A9 delivers immediate checkmate
        }
        board = XiangqiBoard(state, side_to_move=RED)
        engine = AlphaBetaEngine()
        result = engine.search(board, depth=2)

        self.assertIsNotNone(result.best_move)
        self.assertEqual(result.best_move.piece, "RR2")
        self.assertEqual(result.best_move.to_pos, "A9")
        self.assertEqual(result.score, MATE_SCORE - 1)

    def test_mate_in_two_detection(self):
        """
        Engine should find a forced 2-move checkmate (3 plies):
        RR1 at B2 traps BG_ on rank A, while RR2 at C2 is blocked by RR1 on file 2.
        BS1 at D1 prevents a 1-move stalemate, forcing Red to unblock file 2 on move 1
        and deliver checkmate with RR2 on move 2.
        """
        state = {
            "RG_": "J4",
            "BG_": "A5",
            "BS1": "D1",
            "RR1": "B2",
            "RR2": "C2",
        }
        board = XiangqiBoard(state, side_to_move=RED)
        engine = AlphaBetaEngine()

        # Search at depth 3 (Red move 1 -> Black reply -> Red move 2 mate)
        result1 = engine.search(board, depth=3)
        self.assertIsNotNone(result1.best_move)
        self.assertEqual(result1.score, MATE_SCORE - 3)

        # Execute Red's first move
        board.make_move(result1.best_move)
        self.assertFalse(board.is_game_over()[0])

        # Black plays its best defensive reply
        black_reply = engine.search(board, depth=2).best_move
        self.assertIsNotNone(black_reply)
        board.make_move(black_reply)

        # Red executes the mating move
        result2 = engine.search(board, depth=1)
        self.assertIsNotNone(result2.best_move)
        board.make_move(result2.best_move)

        is_over, winner = board.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, RED)

    def test_zobrist_hash_consistency_and_transposition(self):
        """Zobrist hash should be identical when reaching the same state via different move orders."""
        board1 = XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)
        board2 = XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)
        engine = AlphaBetaEngine()

        # Move order 1: Red RC1 H2->H5, Black BC1 C2->C5, Red RA1 J4->I5
        board1.make_move(Move("RC1", pos_to_coords("H2"), pos_to_coords("H5")))
        board1.make_move(Move("BC1", pos_to_coords("C2"), pos_to_coords("C5")))
        board1.make_move(Move("RA1", pos_to_coords("J4"), pos_to_coords("I5")))

        # Move order 2: Red RA1 J4->I5, Black BC1 C2->C5, Red RC1 H2->H5
        board2.make_move(Move("RA1", pos_to_coords("J4"), pos_to_coords("I5")))
        board2.make_move(Move("BC1", pos_to_coords("C2"), pos_to_coords("C5")))
        board2.make_move(Move("RC1", pos_to_coords("H2"), pos_to_coords("H5")))

        h1 = engine.hasher.compute_hash(board1)
        h2 = engine.hasher.compute_hash(board2)
        self.assertEqual(h1, h2)

        # Undoing all moves restores original hash
        for _ in range(3):
            board1.undo_move()
        self.assertEqual(
            engine.hasher.compute_hash(board1),
            engine.hasher.compute_hash(XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)),
        )

    def test_killer_and_history_move_ordering(self):
        """TT move, Captures, Killer moves, and History moves should be ordered in strict priority."""
        m_quiet_plain = Move("RS1", pos_to_coords("G1"), pos_to_coords("F1"))
        m_history = Move("RH1", pos_to_coords("J2"), pos_to_coords("H3"))
        m_killer2 = Move("RE1", pos_to_coords("J3"), pos_to_coords("H5"))
        m_killer1 = Move("RA1", pos_to_coords("J4"), pos_to_coords("I5"))
        m_capture = Move("RC1", pos_to_coords("H2"), pos_to_coords("A2"), captured="BH1")
        m_tt = Move("RR1", pos_to_coords("J1"), pos_to_coords("I1"))

        killers = [m_killer1, m_killer2]
        history = {(m_history.piece, m_history.from_rc, m_history.to_rc): 500}

        ordered = order_moves(
            [m_quiet_plain, m_history, m_killer2, m_killer1, m_capture, m_tt],
            tt_move=m_tt,
            killers=killers,
            history_table=history,
        )
        self.assertEqual(
            ordered,
            [m_tt, m_capture, m_killer1, m_killer2, m_history, m_quiet_plain],
        )

    def test_iterative_deepening_and_tt_speedup(self):
        """Iterative deepening with TT should find the best move and record TT hits."""
        state = {
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
        board = XiangqiBoard(state, side_to_move=RED)
        engine = AlphaBetaEngine(use_tt=True)
        res = engine.iterative_deepening_search(board, max_depth=3, time_limit=5.0)

        self.assertIsNotNone(res.best_move)
        self.assertEqual(res.depth_reached, 3)
        self.assertGreater(res.tt_hits, 0)


if __name__ == "__main__":
    unittest.main()