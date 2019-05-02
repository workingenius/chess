import itertools
from enum import Enum
from string import ascii_lowercase


from errors import BaseMovementError


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
                ab = piece_abbr(p)
                ab_lst.append(ab)
            return ' '.join(ab_lst)

        def piece_abbr(p=None):
            if p is None:
                return '..'
            else:
                return p.job.value[0] + p.camp.value[-1]

        lines = []
        for j in range(7, -1, -1):
            line = format_line(j)
            lines.append(line)

        return '\n'.join(lines)


class BasicMovement(object):
    """movement that don't take chess rule into account"""

    def __init__(self, frm, to=None, capture=None, replace=None):
        self.frm = frm

        if to is None and capture is None:
            raise BaseMovementError('either move to or capture should given')

        # two arguments for move from and move to respectively,
        # for the sake of "En passant"
        self.to = to
        self.capture = capture

        # replace is designed for pawn promotion
        self.replace = replace

    def apply_to(self, chess):
        if chess.square_to_piece.get(self.frm) is None:
            raise BaseMovementError('{} is empty so nothing to move'.format(self.frm.format()))

        if self.to and chess.square_to_piece.get(self.to):
            raise BaseMovementError('{} is occupied by another piece'.format(self.to.format()))

        if self.capture and chess.square_to_piece.get(self.capture) is None:
            raise BaseMovementError('{} is empty so nothing to capture'.format(self.capture.format()))

        if self.replace and self.replace in chess.piece_to_square:
            raise BaseMovementError('the piece to replace is on the board: {}'.format(self.replace.format()))

        pie = chess.remove(square=self.frm)

        if self.capture:
            chess.remove(square=self.capture)

        chess.put(square=self.to or self.capture, piece=self.replace or pie)


class DoubleMovement(BasicMovement):
    """allow two basic movement happens together. designed for castling"""

    # noinspection PyMissingConstructor
    def __init__(self, mv_a, mv_b):
        self.a = mv_a
        self.b = mv_b

    def apply_to(self, chess):
        self.a.apply_to(chess)
        self.b.apply_to(chess)


class Movement(BasicMovement):
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


class Castling(Movement):
    """castling, a special movement"""

    # noinspection PyMissingConstructor
    def __init__(self, camp, short=True):
        self.camp = camp
        self.short = bool(short)
        self.long = not self.short

    def apply_to(self, chess):
        def s(name):
            return Square.by_name(name)

        if self.camp == Camp.A and self.short:
            dm = DoubleMovement(
                BasicMovement(frm=s('e1'), to=s('g1')),
                BasicMovement(frm=s('h1'), to=s('f1'))
            )
        elif self.camp == Camp.A and self.long:
            dm = DoubleMovement(
                BasicMovement(frm=s('e1'), to=s('c1')),
                BasicMovement(frm=s('a1'), to=s('d1'))
            )
        elif self.camp == Camp.B and self.short:
            dm = DoubleMovement(
                BasicMovement(frm=s('e8'), to=s('g8')),
                BasicMovement(frm=s('h8'), to=s('f8'))
            )
        elif self.camp == Camp.A and self.long:
            dm = DoubleMovement(
                BasicMovement(frm=s('e8'), to=s('c8')),
                BasicMovement(frm=s('a8'), to=s('d8'))
            )
        else:
            raise ValueError('cant touch here')

        return dm.apply_to(chess)

    def format(self):
        return 'O-O' if self.short else 'O-O-O'


class Player(object):
    def __call__(self, chess, last_movement=None):
        pass

    @property
    def camp(self):
        pass


def play(player_a, player_b):
    chess = Chess.setup()
    last_movement = None

    for player in itertools.cycle([player_a, player_b]):
        player(chess, last_movement).apply_to(chess)
        if chess.end():
            return chess.result()
