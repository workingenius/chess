import unittest
from chess import play, PlayerByManual, Checkmate, Camp


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


if __name__ == '__main__':
    unittest.main()
