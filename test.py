import unittest
from chess import play, PlayerByManual, Checkmate, Camp, Chess, Job, Square, Movement, validate_movement, RuleBroken


class Test(unittest.TestCase):
    def test_checkmate(self):
        self.assertEqual(

            Checkmate(winner=Camp.A),

            play(*PlayerByManual.pair_by_manual_text('''
                e2-e4  # A
                e7-e5  # B
                f1-c4  # A
                f8-c5  # B
                d1-f3  # A
                b8-a6  # B
                f3xf7  # A
            '''))

        )

    def test_move_out_of_board(self):
        chess = Chess.setup(initial=[
            [Camp.A, Job.CASTLE, Square.by_name('a1')]
        ])
        mv = Movement(frm=Square.by_name('a1'), to=Square.by_name('a0'))

        with self.assertRaises(RuleBroken):
            validate_movement(chess, mv)


if __name__ == '__main__':
    unittest.main()
