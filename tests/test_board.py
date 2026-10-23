import unittest
from src.ai.board import (
    BLACK,
    INITIAL_BOARD_STATE,
    RED,
    Move,
    XiangqiBoard,
    coords_to_pos,
    pos_to_coords,
)


class TestXiangqiBoard(unittest.TestCase):
    def test_board_state_roundtrip(self):
        """Ensure converting from and back to game.py's BOARD_STATE preserves every piece."""
        board = XiangqiBoard(INITIAL_BOARD_STATE)
        exported = board.to_board_state()
        self.assertEqual(exported, INITIAL_BOARD_STATE)

    def test_initial_legal_move_count(self):
        """Standard Xiangqi opening position has exactly 44 legal moves for Red and Black."""
        board = XiangqiBoard(INITIAL_BOARD_STATE, side_to_move=RED)
        red_moves = board.get_legal_moves(RED)
        black_moves = board.get_legal_moves(BLACK)
        self.assertEqual(len(red_moves), 44)
        self.assertEqual(len(black_moves), 44)

    def test_make_and_undo_move(self):
        """Ensure make_move and undo_move restore exact board state including captures."""
        board = XiangqiBoard(INITIAL_BOARD_STATE)
        # Red Cannon RC1 at H2 captures Black Horse BH1 at A2 by jumping BC1 at C2
        capture_move = Move("RC1", pos_to_coords("H2"), pos_to_coords("A2"), captured="BH1")
        board.make_move(capture_move)
        self.assertEqual(board.to_board_state()["RC1"], "A2")
        self.assertEqual(board.to_board_state()["BH1"], "")
        self.assertEqual(board.side_to_move, BLACK)

        board.undo_move()
        self.assertEqual(board.to_board_state(), INITIAL_BOARD_STATE)
        self.assertEqual(board.side_to_move, RED)

    def test_horse_leg_blocking(self):
        """Horse cannot jump if the orthogonally adjacent square ('horse leg') is occupied."""
        custom_state = {
            "RG_": "J5",
            "BG_": "A5",
            "RH1": "E5",
            "RS1": "D5",  # Blocks downward jumps (C4, C6)
        }
        board = XiangqiBoard(custom_state, side_to_move=RED)
        horse_targets = {m.to_pos for m in board.get_legal_moves(RED) if m.piece == "RH1"}
        self.assertNotIn("C4", horse_targets)
        self.assertNotIn("C6", horse_targets)
        self.assertIn("G4", horse_targets)
        self.assertIn("G6", horse_targets)

    def test_elephant_eye_and_river(self):
        """Elephant cannot cross the river (row F -> E) or jump over an occupied elephant eye."""
        custom_state = {
            "RG_": "J5",
            "BG_": "A5",
            "RE1": "F3",  # On Red's riverbank (row F)
            "RS1": "G2",  # Blocks eye toward H1
        }
        board = XiangqiBoard(custom_state, side_to_move=RED)
        elephant_targets = {m.to_pos for m in board.get_legal_moves(RED) if m.piece == "RE1"}
        self.assertNotIn("H1", elephant_targets)  # Eye blocked by RS1 at G2
        self.assertIn("H5", elephant_targets)     # Unblocked on own side
        self.assertNotIn("D1", elephant_targets)  # Cannot cross river into row D
        self.assertNotIn("D5", elephant_targets)  # Cannot cross river into row D

    def test_soldier_river_crossing(self):
        """Soldier only moves forward before river, and forward + sideways after crossing river."""
        custom_state = {
            "RG_": "J5",
            "BG_": "A4",  # Offset general so column 5 is not restricted by Flying General
            "RS1": "F3",  # Before river (Red side) -> can only advance forward to E3
            "RS2": "E7",  # Across river (Black side) -> can advance to D7 or sidestep to E6, E8
        }
        board = XiangqiBoard(custom_state, side_to_move=RED)
        rs1_targets = {m.to_pos for m in board.get_legal_moves(RED) if m.piece == "RS1"}
        rs2_targets = {m.to_pos for m in board.get_legal_moves(RED) if m.piece == "RS2"}
        self.assertEqual(rs1_targets, {"E3"})
        self.assertEqual(rs2_targets, {"D7", "E6", "E8"})

    def test_flying_general_prohibition(self):
        """A piece serving as the sole screen between two Generals cannot move off the file."""
        custom_state = {
            "RG_": "J5",
            "BG_": "A5",
            "RR1": "E5",  # Sole piece on column 5 between RG_ and BG_
        }
        board = XiangqiBoard(custom_state, side_to_move=RED)
        chariot_targets = {m.to_pos for m in board.get_legal_moves(RED) if m.piece == "RR1"}
        # RR1 can only move along column 5 (e.g., I5, H5, ..., A5), never horizontally to E4/E6
        for pos in chariot_targets:
            self.assertEqual(pos[1], "5")

    def test_checkmate_detection(self):
        """Verify is_game_over detects when a player is checkmated."""
        custom_state = {
            "RG_": "J5",
            "BG_": "A5",
            "RR1": "A1",  # Delivers check along rank A
            "RR2": "B1",  # Controls rank B so BG_ cannot escape up to B5
        }
        board = XiangqiBoard(custom_state, side_to_move=BLACK)
        self.assertTrue(board.is_in_check(BLACK))
        is_over, winner = board.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, RED)

    def test_direct_check_detection_all_piece_types(self):
        """Verify fast O(1) is_in_check accurately detects checks from Cannon, Horse, and Soldier."""
        # 1. Cannon check with 1 screen (in check) vs 2 screens (not in check)
        cannon_1_screen = {"RG_": "J5", "BG_": "A4", "RS1": "H5", "BC1": "C5"}
        cannon_2_screens = {"RG_": "J5", "BG_": "A4", "RS1": "H5", "BS1": "E5", "BC1": "C5"}
        self.assertTrue(XiangqiBoard(cannon_1_screen).is_in_check(RED))
        self.assertFalse(XiangqiBoard(cannon_2_screens).is_in_check(RED))

        # 2. Horse check with unblocked leg (in check) vs blocked leg (not in check)
        # BH1 at H3 jumping to J4 has its orthogonal leg at I3
        horse_unblocked = {"RG_": "J4", "BG_": "A5", "BH1": "H3"}
        horse_blocked = {"RG_": "J4", "BG_": "A5", "BH1": "H3", "RS1": "I3"}  # I3 blocks the jump leg
        self.assertTrue(XiangqiBoard(horse_unblocked).is_in_check(RED))
        self.assertFalse(XiangqiBoard(horse_blocked).is_in_check(RED))

        # 3. Soldier check (forward and sideways in palace)
        soldier_forward = {"RG_": "I5", "BG_": "A4", "BS1": "H5"}  # BS1 moves up from H5 -> I5
        soldier_sideways = {"RG_": "I5", "BG_": "A4", "BS1": "I4"}
        soldier_behind = {"RG_": "I5", "BG_": "A4", "BS1": "J5"}   # BS1 at J5 cannot move backward to I5
        self.assertTrue(XiangqiBoard(soldier_forward).is_in_check(RED))
        self.assertTrue(XiangqiBoard(soldier_sideways).is_in_check(RED))
        self.assertFalse(XiangqiBoard(soldier_behind).is_in_check(RED))


if __name__ == "__main__":
    unittest.main()