import itertools
from enum import Enum
from string import ascii_lowercase
from typing import List

from errors import BaseMovementError

BOARD_SIZE = 8


class Color(Enum):
    WHITE = 'WHITE'
    BLACK = 'BLACK'


class Camp(Enum):
    A = 'CAMP_A'
    B = 'CAMP_B'


class Job(Enum):
    """which kind a piece is"""

    KING = 'KING'
    QUEEN = 'QUEEN'
    BISHOP = 'BISHOP'
    KNIGHT = 'KNIGHT'
    CASTLE = 'CASTLE'
    PAWN = 'PAWN'


class Square(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def color(self):
        return Color.BLACK if (self.x + self.y) % 2 == 0 else Color.WHITE

    @property
    def name(self):
        return ascii_lowercase[self.x] + str(self.y + 1)

    def format(self):
        return str(self)

    def __str__(self):
        return 'square {}'.format(self.name)

    @classmethod
    def by_name(cls, name):
        column = name[0]
        line = name[1]
        x = ascii_lowercase.index(column)
        y = int(line) - 1
        return cls(x, y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, Square) and self.x == other.x and self.y == other.y

    def is_ending_line(self, camp, index=0):
        if camp == Camp.A:
            return self.y == (7 - index)
        else:
            return self.y == (0 + index)

    def is_starting_line(self, camp, index=0):
        if camp == Camp.A:
            return self.y == (0 + index)
        else:
            return self.y == (7 - index)

    @staticmethod
    def is_adjacent(sq1, sq2, horizontally=False):
        if horizontally:
            return sq1.y == sq2.y and abs(sq1.x - sq2.x) == 1
        else:
            return max(abs(sq1.x - sq2.x), abs(sq1.y - sq2.y)) == 1


class Piece(object):
    def __init__(self, camp, job):
        self.camp = camp
        self.job = job
        self.square = None

    def __str__(self):
        if self.square:
            return '{} of {} at {}'.format(self.job.value, self.camp.value, self.square)
        else:
            return '{} of {}'.format(self.job.value, self.camp.value)

    def format(self):
        return str(self)


class Chess(object):
    @classmethod
    def setup(cls):

        def s(name):
            return Square.by_name(name)

        c = Chess()

        locations = [
            [Camp.A, Job.CASTLE, s('a1')],
            [Camp.A, Job.KNIGHT, s('b1')],
            [Camp.A, Job.BISHOP, s('c1')],
            [Camp.A, Job.QUEEN, s('d1')],
            [Camp.A, Job.KING, s('e1')],
            [Camp.A, Job.BISHOP, s('f1')],
            [Camp.A, Job.KNIGHT, s('g1')],
            [Camp.A, Job.CASTLE, s('h1')],

            [Camp.A, Job.PAWN, s('a2')],
            [Camp.A, Job.PAWN, s('b2')],
            [Camp.A, Job.PAWN, s('c2')],
            [Camp.A, Job.PAWN, s('d2')],
            [Camp.A, Job.PAWN, s('e2')],
            [Camp.A, Job.PAWN, s('f2')],
            [Camp.A, Job.PAWN, s('g2')],
            [Camp.A, Job.PAWN, s('h2')],

            [Camp.B, Job.CASTLE, s('a8')],
            [Camp.B, Job.KNIGHT, s('b8')],
            [Camp.B, Job.BISHOP, s('c8')],
            [Camp.B, Job.QUEEN, s('d8')],
            [Camp.B, Job.KING, s('e8')],
            [Camp.B, Job.BISHOP, s('f8')],
            [Camp.B, Job.KNIGHT, s('g8')],
            [Camp.B, Job.CASTLE, s('h8')],

            [Camp.B, Job.PAWN, s('a7')],
            [Camp.B, Job.PAWN, s('b7')],
            [Camp.B, Job.PAWN, s('c7')],
            [Camp.B, Job.PAWN, s('d7')],
            [Camp.B, Job.PAWN, s('e7')],
            [Camp.B, Job.PAWN, s('f7')],
            [Camp.B, Job.PAWN, s('g7')],
            [Camp.B, Job.PAWN, s('h7')],
        ]

        for loc in locations:
            c.put(piece=Piece(camp=loc[0], job=loc[1]), square=loc[2])

        return c

    def __init__(self):
        self.square_to_piece = {}
        self.piece_to_square = {}

        for i in range(8):
            for j in range(8):
                self.square_to_piece[Square(i, j)] = None

        self.turn = None  # who's turn
        self.history = []

    def put(self, piece, square):
        assert piece not in self.piece_to_square
        self.square_to_piece[square] = piece
        self.piece_to_square[piece] = square

    def remove(self, piece=None, square=None):
        if piece is None and square is None:
            raise ValueError('either piece or square should be given')

        if piece is not None and square is not None:
            if self.piece_to_square.get(piece) != square or \
                    self.square_to_piece.get(square) != piece:
                raise ValueError('piece and square maps wrong')

        # remove piece by piece and return square
        if piece:
            if piece not in self.piece_to_square:
                raise ValueError('piece is not on board and cant remove')

            sq = self.piece_to_square.pop(piece)
            self.square_to_piece[sq] = None
            return sq

        # remove piece by square and return piece
        elif square:
            pi = self.square_to_piece.get(square)
            if pi:
                self.square_to_piece[square] = None
                self.piece_to_square.pop(pi)
                return pi

    def format(self):
        """a simple way to visualize"""

        def format_line(y):
            ab_lst = []
            for i in range(8):
                sq = Square(i, y)
                p = self.square_to_piece[sq]
                ab = piece_abbr(sq, p)
                ab_lst.append(ab)
            return ' '.join(ab_lst)

        def job_abbr(job):
            if job == Job.KING:
                return 'Z'
            else:
                return job.value[0].upper()

        def piece_abbr(sq, p=None):
            if p is None:
                return '..' if sq.color == Color.WHITE else '||'
            else:
                return job_abbr(p.job) + p.camp.value[-1].lower()

        lines = []
        for j in range(7, -1, -1):
            line = format_line(j)
            lines.append(line)

        return '\n'.join(lines)


class Movement(object):
    """movement that don't take chess rule into account"""

    def __init__(self, frm, to=None, capture=None, replace=None, sub_movement: 'Movement' = None):
        self.frm = frm

        if to is None and capture is None:
            raise BaseMovementError('either move to or capture should given')

        # two arguments for move from and move to respectively,
        # for the sake of "En passant"
        self.to = to or capture
        self.capture = capture

        # replace is designed for pawn promotion
        self.replace = replace

        self.sub_movement = sub_movement

    def apply_to(self, chess):
        if chess.square_to_piece.get(self.frm) is None:
            raise BaseMovementError('{} is empty so nothing to move'.format(self.frm.format()))

        if not self.capture and self.to and chess.square_to_piece.get(self.to):
            raise BaseMovementError('{} is occupied by another piece'.format(self.to.format()))

        if self.capture and chess.square_to_piece.get(self.capture) is None:
            raise BaseMovementError('{} is empty so nothing to capture'.format(self.capture.format()))

        if self.replace and self.replace in chess.piece_to_square:
            raise BaseMovementError('the piece to replace is on the board: {}'.format(self.replace.format()))

        pie = chess.remove(square=self.frm)

        if self.capture:
            chess.remove(square=self.capture)

        chess.put(square=self.to or self.capture, piece=self.replace or pie)

        if self.sub_movement:
            chess = self.sub_movement.apply_to(chess)

        return chess

    @property
    def name(self):
        if self.capture:
            return '{}x{}'.format(self.frm.name, self.capture.name)
        elif self.to:
            return '{}-{}'.format(self.frm.name, self.to.name)
        else:
            raise ValueError('cant get here')

    def format(self):
        return 'movement {}'.format(self.name)

    @classmethod
    def by_name(cls, name):
        """notice: special movements like promotion, En passant, castling cant be created by name"""

        f, t, c = None, None, None
        if '-' in name:
            f, t = name.split('-')
        elif 'x' in name:
            f, c = name.split('x')

        return cls(frm=Square.by_name(f),
                   to=Square.by_name(t) if t else None,
                   capture=Square.by_name(c) if c else None)


_m = Movement.by_name


class Direction(object):
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

    @property
    def is_zero(self):
        return self.x == 0 and self.y == 0

    @property
    def is_vertical(self):
        return self.x == 0 and self.y != 0

    @property
    def is_horizontal(self):
        return self.x != 0 and self.y == 0

    @property
    def is_north(self):
        return self.x == 0 and self.y > 0

    @property
    def is_south(self):
        return self.x == 0 and self.y < 0

    @property
    def is_west(self):
        return self.x < 0 and self.y == 0

    @property
    def is_east(self):
        return self.x > 0 and self.y == 0

    @property
    def is_north_west(self):
        return self.x < 0 and self.y > 0

    @property
    def is_north_east(self):
        return self.x > 0 and self.y > 0

    @property
    def is_south_west(self):
        return self.x < 0 and self.y < 0

    @property
    def is_south_east(self):
        return self.x > 0 and self.y < 0

    def is_forward(self, camp, just=True):
        if just:
            if camp == Camp.A:
                return self.is_north
            else:
                return self.is_south
        else:
            if camp == Camp.A:
                return self.y > 0
            else:
                return self.y < 0

    def is_backward(self, camp, just=True):
        if just:
            if camp == Camp.A:
                return self.is_south
            else:
                return self.is_north
        else:
            if camp == Camp.A:
                return self.y > 0
            else:
                return self.y < 0


class MovePath(object):
    """a vertical, horizontal or diagonal move path"""

    class InvalidPath(ValueError):
        pass

    def __init__(self, start, end):
        self.start: Square = start
        self.end: Square = end

        # squares that move passes, starting point and ending point are not included
        self.passes: List[Square] = None

        # move distance
        self.dis: int = None

        # direction
        self.direction: Direction = Direction(x=(end.x - start.x), y=(end.y - start.y))

        self.initial_calc()

    def moved(self):
        return self.start != self.end

    def __bool__(self):
        return self.moved()

    def initial_calc(self):
        passes = []
        dis = 0

        if self.moved():
            start = self.start
            end = self.end

            if start.x == end.x:
                # vertical path
                passes = [Square(x=start.x, y=i) for i in self.from_to(start.y, end.y)]

            elif start.y == end.y:
                # horizontal path
                passes = [Square(x=i, y=start.y) for i in self.from_to(start.x, end.x)]

            elif abs(start.x - end.x) == abs(start.y - end.y):
                # diagonal path
                x_lst = self.from_to(start.x, end.x)
                y_lst = self.from_to(start.y, end.y)
                passes = [Square(x=x, y=y) for x, y in zip(x_lst, y_lst)]

            else:
                raise self.InvalidPath((self.start, self.end))

            dis = len(passes) + 1

        self.passes = passes
        self.dis = dis

    @staticmethod
    def from_to(a, b):
        if a < b:
            return list(range(a + 1, b))
        elif a > b:
            return list(range(a - 1, b, -1))


class RuleOK(object):
    ok = True

    def __bool__(self):
        return True


class RuleBroken(Exception):
    ok = False

    def __bool__(self):
        return False


def validate_movement(chess, mv: Movement):
    # basic checks

    sq, pi = mv.frm, chess.square_to_piece.get(mv.frm)
    if pi is None:
        return RuleBroken('There is not a piece')

    if pi.camp != chess.turn:
        return RuleBroken('You can\'t move your partner\'s piece')

    if mv.capture and not chess.square_to_piece.get(mv.capture):
        return RuleBroken('There\'s nothing to capture')

    # check move path

    if pi.job == Job.KNIGHT:
        # only knight has no move path
        pass

    else:
        try:
            path = MovePath(start=mv.frm, end=mv.to)
        except MovePath.InvalidPath:
            return RuleBroken('No piece can fly')

        if not path.moved():
            return RuleBroken('The piece doesnt move')

        for pss_sq in path.passes:
            _pi = chess.square_to_piece.get(pss_sq)
            if _pi:
                return RuleBroken('Another piece is in the way')

        target_sq = path.end
        target_pi = chess.square_to_piece.get(target_sq)
        if target_pi:
            if target_pi.camp == pi.camp:
                return RuleBroken('You can\'t capture piece of your own camp')

    # piece related checks

    def goto_capture():
        if mv.capture:
            if mv.to != mv.capture:
                return RuleBroken('You cant kill piece by magic')

    def check_replace():
        if mv.replace:
            return RuleBroken('Only pawns have change to promote')

    def check_castle():
        di = path.direction
        if not (di.is_vertical or di.is_horizontal):
            return RuleBroken('Castle can only move horizontally or vertically')

        return goto_capture() or check_replace() or RuleOK()

    def check_queen():
        # no need to check it's direction
        # because queen can go in any direction as long as it does not fly
        return goto_capture() or check_replace() or RuleOK()

    def check_bishop():
        di = path.direction
        if not (di.is_south_east or di.is_south_west or di.is_north_east or di.is_north_west):
            return RuleBroken('Bishop can only move diagonally')

        return goto_capture() or check_replace() or RuleOK()

    def check_pawn():

        # check promotion
        if mv.replace:
            if not mv.to.is_ending_line(chess.turn):
                return RuleBroken('Pawns can promote only when they reach the ending line')

            if mv.replace.job == Job.KING:
                return RuleBroken('A pawn can not promote to a king')

        # check movement
        if not path.direction.is_forward(camp=chess.turn, just=False):
            return RuleBroken('A pawn can only proceed')

        if path.dis > 2:
            return RuleBroken('The pawn moves too fast')

        if path.direction.is_forward(camp=chess.turn, just=True):
            # is proceeding

            # check starting charge
            if path.dis == 2:
                if mv.frm.is_starting_line(camp=chess.turn, index=1):
                    pass
                else:
                    return RuleBroken('A pawn can only charge from his beginning point')

            if mv.capture:
                if chess.square_to_piece.get(mv.capture):
                    # an enemy is ahead
                    return RuleBroken('Pawns cant attack enemy in front of him')
                else:
                    return RuleBroken('Pawns cant attack when proceeding')

        else:
            # is attacking
            if path.dis == 2:
                return RuleBroken('A pawn can only attack enemy in diagonal direction by one square')

            # TODO: check en passant
            return goto_capture() or RuleOK()

        return RuleOK()

    def check_knight():
        if {abs(mv.to.x - mv.frm.x), abs(mv.to.y - mv.frm.y)} != {1, 2}:
            return RuleBroken('Knight can only move in L shape')

        return goto_capture() or check_replace() or RuleOK()

    def check_king():
        if not Square.is_adjacent(mv.frm, mv.to):
            return RuleBroken('The king can only move to his adjacent square')

        return goto_capture() or check_replace() or RuleOK()

    check_map = {
        Job.BISHOP: check_bishop,
        Job.QUEEN: check_queen,
        Job.PAWN: check_pawn,
        Job.KNIGHT: check_knight,
        Job.KING: check_king,
        Job.CASTLE: check_castle
    }

    check = check_map[pi.job]()
    if not check.ok:
        return check

    # TODO: king related checks

    return RuleOK()


class Player(object):
    def __init__(self, camp):
        self._camp = camp

    def __call__(self, chess):
        cmd = input('{} turn >> '.format(self.camp)).strip().lower()
        return _m(cmd)

    @property
    def camp(self):
        return self._camp


def play(player_a, player_b):
    chess = Chess.setup()

    for player in itertools.cycle([player_a, player_b]):
        chess.turn = player.camp

        print()
        print(chess.format())

        mv = player(chess)
        assert isinstance(mv, Movement)
        rule_check = validate_movement(chess, mv)

        while not rule_check.ok:
            print(rule_check)
            print()

            mv = player(chess)
            assert isinstance(mv, Movement)
            rule_check = validate_movement(chess, mv)

        chess = mv.apply_to(chess)
        # r = check_result(chess)
        # if r:
        #     return r


if __name__ == '__main__':
    play(Player(Camp.A), Player(Camp.B))
