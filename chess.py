import itertools
from enum import Enum
from functools import wraps
from string import ascii_lowercase
from typing import Iterator, Optional, List, Tuple

from errors import BaseMovementError

BOARD_SIZE = 8


class Color(Enum):
    WHITE = 'WHITE'
    BLACK = 'BLACK'


class Camp(Enum):
    A = 'CAMP_A'
    B = 'CAMP_B'

    @property
    def another(self):
        if self == Camp.A:
            return Camp.B
        elif self == Camp.B:
            return Camp.A


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

    def __repr__(self):
        return 'Square(x={s.x}, y={s.y})'.format(s=self)

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
            return self.y == (BOARD_SIZE - 1 - index)
        else:
            return self.y == (0 + index)

    def is_starting_line(self, camp, index=0):
        if camp == Camp.A:
            return self.y == (0 + index)
        else:
            return self.y == (BOARD_SIZE - 1 - index)

    @staticmethod
    def is_adjacent(sq1, sq2, horizontally=False):
        if horizontally:
            return sq1.y == sq2.y and abs(sq1.x - sq2.x) == 1
        else:
            return max(abs(sq1.x - sq2.x), abs(sq1.y - sq2.y)) == 1

    def __add__(self, delta: 'Delta'):
        return self.__class__(x=(self.x + delta.x), y=(self.y + delta.y))

    def __sub__(self, sq: 'Square'):
        return Delta(x=(self.x - sq.x), y=(self.y - sq.y))

    def is_on_board(self):
        return 0 <= self.x < BOARD_SIZE and 0 <= self.y < BOARD_SIZE


def _on_board(sq_lst):
    return filter(lambda x: x.is_on_board(), sq_lst)


def within_board(f):
    @wraps(f)
    def ff(*args, **kwargs):
        return _on_board(f(*args, **kwargs))

    return ff


def _has_enemy(chess, camp, at):
    return chess.square_to_piece.get(at) and chess.square_to_piece.get(at).camp != camp


def _has_alias(chess, camp, at):
    return chess.square_to_piece.get(at) and chess.square_to_piece.get(at).camp == camp


def _has_piece(chess, at):
    return bool(chess.square_to_piece.get(at))


class Piece(object):
    @property
    def job(self):
        raise NotImplementedError

    def __init__(self, camp):
        self.camp = camp
        self.square = None

    def __str__(self):
        if self.square:
            return '{} of {} at {}'.format(self.job.value, self.camp.value, self.square)
        else:
            return '{} of {}'.format(self.job.value, self.camp.value)

    def __repr__(self):
        return '{cls}({cm})'.format(cls=self.__class__.__name__, cm=self.camp)

    def format(self):
        return str(self)

    def attack_lst(self, chess) -> Iterator[Square]:
        """generate all squares it is able to capture in the next movement,
        even if there's not a piece or the piece is of same camp"""
        raise NotImplementedError

    def movement_lst(self, chess) -> Iterator['Movement']:
        """generate all movements launched by the piece"""
        raise NotImplementedError

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        """if the movement <mv> about the job is valid. If not, tell reason"""
        raise NotImplementedError

    # rule helpers

    def assert_valid_capture(self, chess, capture):
        ca = chess.square_to_piece.get(capture)

        if ca is None:
            raise RuleBroken('There is no piece to capture')

        if ca.camp != self.camp:
            pass
        else:
            raise RuleBroken('You can not capture piece of your own camp')

    def assert_valid_move(self, chess, move: Square):
        if _has_enemy(chess, self.camp, at=move) or _has_alias(chess, self.camp, at=move):
            raise RuleBroken('You can not move to a square with piece')

    def assert_clear_path(self, chess, path: Iterator[Square]):
        for sq in path:
            if _has_piece(chess, at=sq):
                raise RuleBroken('You can not move over other pieces')

    def move_to_or_capture(self, chess, frm, to):
        if _has_enemy(chess, camp=self.camp, at=to):
            return Movement(frm=frm, capture=to)
        elif _has_alias(chess, camp=self.camp, at=to):
            return
        else:
            return Movement(frm=frm, to=to)

    def check_target(self, chess, frm, to_lst):
        for to in to_lst:
            mv = self.move_to_or_capture(chess, frm, to)
            if mv:
                yield mv

    def extend_till_hidden(self, chess, square, direction):
        hidden = False

        for dis in range(1, BOARD_SIZE):
            if hidden:
                break

            delta = direction * dis
            to = square + delta

            if _has_piece(chess, at=to):
                hidden = True
                yield to

            else:
                yield to


class PKing(Piece):
    job = Job.KING

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        # squares surrounding it in all eight direction on board
        sq = chess.piece_to_square.get(self)
        for di in ALL_DIRECTIONS:
            yield sq + di

    def movement_lst(self, chess) -> Iterator['Movement']:
        # regular moves
        #   squares surrounding it in all eight direction on board, able to move there and capture a enemy
        # castling
        #   positive condition:
        #     1. king has not moved
        #     2. the castle involved has not moved
        #     3. no any other pieces in the way
        #     4. squares in the way is not currently attacked by any enemy piece
        #   movement:
        #     1. king move toward the castle by two square
        #     2. the castle move toward, over, and right beside king
        frm = chess.piece_to_square.get(self)
        return self.check_target(chess, frm, self.attack_lst(chess))

        # TODO: CASTLING

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        king_rule_caption = 'King only move straight by one square'

        try:
            delta = mv.to - mv.frm
            passes(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            raise RuleBroken(king_rule_caption)

        if delta.dis >= 2:
            raise RuleBroken(king_rule_caption)

        if mv.capture:
            self.assert_valid_capture(chess, mv.capture)
        else:
            self.assert_valid_move(chess, mv.to)

        # TODO: check CASTLING

        return RULE_OK


class PPawn(Piece):
    job = Job.PAWN

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        # one square ahead of columns on both side
        # special case: en passant
        #   If a pawn is beside an enemy pawn who has just charged two square,
        #   he can move diagonally as usual and attack back to capture the enemy pawn

        yield chess.piece_to_square[self] + Delta.as_camp(forward=1, leftward=1, camp=self.camp)
        yield chess.piece_to_square[self] + Delta.as_camp(forward=1, rightward=1, camp=self.camp)

        # TODO: en passant

    def movement_lst(self, chess) -> Iterator['Movement']:
        at = chess.piece_to_square[self]

        def mv_with_promotion(frm, to=None, capture=None):
            target = to or capture

            if target.is_ending_line(camp=self.camp):
                for prom in [
                    cons_piece(self.camp, Job.QUEEN),
                    cons_piece(self.camp, Job.KNIGHT),
                    cons_piece(self.camp, Job.BISHOP),
                    cons_piece(self.camp, Job.CASTLE),
                ]:
                    yield Movement(frm=frm, to=to, capture=capture, replace=prom)

            else:
                yield Movement(frm=frm, to=to, capture=capture)

        # check charge
        if at.is_starting_line(index=1, camp=self.camp):
            d = Delta.as_camp(forward=2)
            to = at + d
            if not in_the_way(chess, passes(at, delta=d)) and not _has_piece(chess, at=to):
                # no need to check for promotion in starting point
                yield Movement(frm=at, to=to)

        # one step forward
        d = Delta.as_camp(forward=1)
        to = at + d
        if not _has_piece(chess, at=to):
            for mv in mv_with_promotion(frm=at, to=to):
                yield mv

        # capture
        for cp in self.attack_lst(chess):
            if _has_enemy(chess, camp=self.camp, at=cp):
                for mv in mv_with_promotion(frm=at, capture=cp):
                    yield mv

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':

        # check promotion
        if mv.replace:
            if not mv.to.is_ending_line(chess.turn):
                raise RuleBroken('Pawns can promote only when they reach the ending line')

            if mv.replace.job == Job.KING:
                raise RuleBroken('A pawn can not promote to a king')

        delta = mv.to - mv.frm

        # check movement
        if not delta.is_forward(camp=self.camp, just=False):
            raise RuleBroken('A pawn can only proceed')

        if delta.is_forward(camp=chess.turn, just=True):
            # is moving ahead

            if delta.dis > 2:
                raise RuleBroken('The pawn moves too fast')

            # check starting charge
            if delta.dis == 2:
                if not mv.frm.is_starting_line(camp=chess.turn, index=1):
                    raise RuleBroken('A pawn can only charge from his beginning point')

                # can't go over piece
                path = passes(start=mv.frm, delta=delta)
                self.assert_clear_path(chess, path)

            if mv.capture:
                raise RuleBroken('Pawns cant attack ahead')
            else:
                self.assert_valid_move(chess, mv.to)

        else:
            # is attacking
            pawn_cap_rule_caption = 'A pawn can only attack enemy in diagonal direction by one square'

            if mv.capture is None:
                raise RuleBroken(pawn_cap_rule_caption)

            if delta.dis != 1:
                raise RuleBroken(pawn_cap_rule_caption)

            self.assert_valid_capture(chess, capture=mv.capture)

            # TODO: check en passant

        return RULE_OK


class PKnight(Piece):
    job = Job.KNIGHT

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for d in L_SHAPES:
            yield chess.piece_to_square[self] + d

    def movement_lst(self, chess) -> Iterator['Movement']:
        return self.check_target(chess, chess.piece_to_square[self], self.attack_lst(chess))

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        delta = mv.to - mv.frm
        if not delta.is_l_shape:
            raise RuleBroken('Knight can only move in l shape')

        if mv.capture:
            self.assert_valid_capture(chess, mv.capture)
        else:
            self.assert_valid_move(chess, mv.to)

        return RULE_OK


class PQueen(Piece):
    job = Job.QUEEN

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in ALL_DIRECTIONS:
            for to in self.extend_till_hidden(chess, chess.piece_to_square[self], dir):
                yield to

    def movement_lst(self, chess) -> Iterator['Movement']:
        return self.check_target(chess, chess.piece_to_square[self], self.attack_lst(chess))

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        # valid direction
        try:
            pss = passes(start=mv.frm, end=mv.to, straight_only=True)
        except InvalidPath:
            raise RuleBroken('Queen can only move straightly')

        # valid moving path
        self.assert_clear_path(chess, pss)

        if mv.capture:
            self.assert_valid_capture(chess, mv.capture)
        else:
            self.assert_valid_move(chess, mv.to)

        return RULE_OK


class PCastle(Piece):
    job = Job.CASTLE

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in HORIZONTAL_AND_VERTICAL_DIRECTIONS:
            for to in self.extend_till_hidden(chess, chess.piece_to_square[self], dir):
                yield to

    def movement_lst(self, chess) -> Iterator['Movement']:
        return self.check_target(chess, chess.piece_to_square[self], self.attack_lst(chess))

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        castle_rule_caption = 'Castle can only move in vertical or horizontal lines'

        # valid direction
        try:
            delta = mv.to - mv.frm
            pss = passes(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            raise RuleBroken(castle_rule_caption)

        if delta.is_horizontal or delta.is_vertical:
            pass
        else:
            raise RuleBroken(castle_rule_caption)

        # valid moving path
        self.assert_clear_path(chess, pss)

        if mv.capture:
            self.assert_valid_capture(chess, mv.capture)
        else:
            self.assert_valid_move(chess, mv.to)

        return RULE_OK


class PBishop(Piece):
    job = Job.BISHOP

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in DIAGONAL_DIRECTIONS:
            for to in self.extend_till_hidden(chess, chess.piece_to_square[self], dir):
                yield to

    def movement_lst(self, chess) -> Iterator['Movement']:
        return self.check_target(chess, chess.piece_to_square[self], self.attack_lst(chess))

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        castle_rule_caption = 'Bishop can only move in diagonal lines'

        # valid direction
        try:
            delta = mv.to - mv.frm
            pss = passes(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            raise RuleBroken(castle_rule_caption)

        if delta.is_diagonal:
            pass
        else:
            raise RuleBroken(castle_rule_caption)

        # valid moving path
        self.assert_clear_path(chess, pss)

        if mv.capture:
            self.assert_valid_capture(chess, mv.capture)
        else:
            self.assert_valid_move(chess, mv.to)

        return RULE_OK


def cons_piece(camp, job):
    return {
        Job.KNIGHT: PKnight,
        Job.PAWN: PPawn,
        Job.KING: PKing,
        Job.QUEEN: PQueen,
        Job.CASTLE: PCastle,
        Job.BISHOP: PBishop
    }[job](camp)


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
            c.put(piece=cons_piece(camp=loc[0], job=loc[1]), square=loc[2])

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

    def pieces(self, camp=None):
        pi_lst = self.piece_to_square.keys()

        if camp:
            pi_lst = filter(lambda p: p.camp == camp, pi_lst)

        return pi_lst

    def find_king(self, camp=None) -> Tuple[Square, Piece]:
        """Find out your king"""

        if camp is None:
            camp = self.turn

        def find_out_king():
            for pi in self.pieces(camp=camp):
                if pi.job == Job.KING:
                    return pi

        ki = find_out_king()

        return self.piece_to_square[ki], ki


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

    def __str__(self):
        return self.format()

    def __repr__(self):
        return str(self)

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


class Delta(object):
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

    @classmethod
    def as_camp(cls, forward=None, backward=None, leftward=None, rightward=None, camp=Camp.A):
        assert not (forward and backward), 'forward and backward cant both exist'
        assert not (leftward and rightward), 'leftward and rightward cant both exist'

        forward = forward or 0
        backward = backward or 0
        leftward = leftward or 0
        rightward = rightward or 0

        if camp == Camp.A:
            x = rightward or -leftward
            y = forward or -backward
        elif camp == Camp.B:
            x = leftward or -rightward
            y = backward or -forward
        else:
            raise ValueError

        return cls(x=x, y=y)

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

    @property
    def is_diagonal(self):
        return self.is_north_west or self.is_north_east or self.is_south_west or self.is_south_east

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

    @property
    def is_l_shape(self):
        """knight move in l shape"""
        return {abs(self.x), abs(self.y)} == {1, 2}

    @property
    def dis(self) -> int:
        return max(abs(self.x), abs(self.y))

    def __mul__(self, other):
        return self.__class__(x=self.x * other, y=self.y * other)

    def __neg__(self):
        return self.__class__(x=-self.x, y=-self.y)

    def moved(self):
        return self.x != 0 or self.y != 0

    def __bool__(self):
        return self.moved()

    @property
    def unit(self):
        return self.__class__(
            x=(self.x // abs(self.x or 1)),
            y=(self.y // abs(self.y or 1))
        )


ALL_DIRECTIONS = [
    Delta(0, 1),  # north
    Delta(1, 1),  # north east
    Delta(1, 0),  # east
    Delta(1, -1),  # south east
    Delta(0, -1),  # south
    Delta(-1, -1),  # south west
    Delta(-1, 0),  # west
    Delta(-1, 1),  # north west
]

HORIZONTAL_AND_VERTICAL_DIRECTIONS = [d for d in ALL_DIRECTIONS if d.is_horizontal or d.is_vertical]

DIAGONAL_DIRECTIONS = [d for d in ALL_DIRECTIONS if d.is_diagonal]

L_SHAPES = [
    Delta(-2, 1),
    Delta(-1, 2),
    Delta(1, 2),
    Delta(2, 1),
    Delta(2, -1),
    Delta(1, -2),
    Delta(-1, -2),
    Delta(-2, -1)
]


class InvalidPath(ValueError):
    pass


def passes(start: Square, end: Square = None, delta: Delta = None, straight_only=True) -> List[Square]:
    if end and delta:
        raise ValueError('either end or delta should be given')

    if end:
        delta = end - start

    err = None

    if delta.x != 0 and delta.y != 0 and abs(delta.x) != abs(delta.y):
        err = InvalidPath((delta.x, delta.y))

    if err:
        if straight_only:
            raise err
        else:
            return []

    unit = delta.unit
    return [start + (unit * i) for i in range(1, delta.dis)]


def in_the_way(chess, pass_way):
    """if some piece is in the way"""

    for sq in pass_way:
        if chess.square_to_piece.get(sq):
            return True
    return False


class RuleStatus(object):
    pass


class RuleOK(RuleStatus):
    ok = True

    def __bool__(self):
        return True


RULE_OK = RuleOK()


class RuleBroken(RuleStatus, Exception):
    ok = False

    def __bool__(self):
        return False


class KingInDanger(RuleBroken):
    pass


def validate_movement(chess, mv: Movement):
    # REWRITING validate_movement

    # 1. You have to move a piece, but not move air
    # 2. The piece you move must be of your own camp, you can't move your partner's piece

    # 3. The movement must comply move rule of the job ... It differs from the job , but generally speaking,
    #   a. If you want to capture another piece, make sure there's one to capture
    #   b. If you capture a piece, it can't be of your own camp

    # 4. If your king is currently in danger, the movement must save your king.
    #     (note: If the king is currently in danger, and none of possible movements is able to save him, it's checkmate
    #     and should be found right after last movement.)
    # 5. And make sure that the movement will not expose your king to danger.

    # In step 3:
    #     Piece.is_valid_movement(self: Piece, chess: Chess, mv: Movement) -> Union[ RuleOk, RuleBroken ]
    #         """Check if a movement is valid according to rule of the job, and return why if broken."""
    #
    # In step 4:
    #     Piece.attacks(self: Piece, chess: Chess, sq: Square) -> bool
    #         """Check if the piece <self> in <chess> is attacking a certain square <sq>"""
    #         This function is needed to check if the king is currently in danger or will be in danger.

    pi: Piece = chess.square_to_piece.get(mv.frm)
    if pi is None:
        return RuleBroken('There is not a piece')

    if pi.camp != chess.turn:
        return RuleBroken('You can\'t move your partner\'s piece')

    try:
        pi.is_valid_movement(chess, mv)

        def king_location():
            if pi.job == Job.KING:
                return mv.to
            else:
                ki_sq, ki_pi = chess.find_king(camp=chess.turn)
                return ki_sq

        king_loc = king_location()

        # traverse piece from other camp
        if any(piece_lst_attacks(chess, sq=king_loc, camp=chess.turn.another)):
            raise KingInDanger('King in danger')

    except RuleBroken as rb:
        return rb

    return RULE_OK


class Player(object):
    def __init__(self, camp):
        self._camp = camp

    def __call__(self, chess):
        try:
            cmd = input('{} turn >> '.format(self.camp)).lower()
        except EOFError:
            cmd = 'resign'

        return self.parse_command(cmd)

    @staticmethod
    def parse_command(cmd):
        # comments after "#"
        if '#' in cmd:
            cmd = cmd.split('#')[0]

        cmd = cmd.lower().strip()

        if cmd == 'resign':
            return ResignRequest()
        else:
            return _m(cmd)

    @property
    def camp(self):
        return self._camp


class PlayerByManual(Player):
    def __init__(self, camp, movement_lst):
        super(PlayerByManual, self).__init__(camp)
        self.movement_lst = movement_lst
        self.cur = 0

    def __call__(self, chess):
        if self.cur < len(self.movement_lst):
            self.cur += 1
            return self.parse_command(self.movement_lst[self.cur - 1])
        else:
            return ResignRequest()

    @classmethod
    def pair_by_manual_text(cls, text):
        cmd_lst = [t.strip() for t in text.split('\n') if t.strip()]
        a_cmd_lst = [cmd for i, cmd in enumerate(cmd_lst) if i % 2 == 0]
        b_cmd_lst = [cmd for i, cmd in enumerate(cmd_lst) if i % 2 == 1]
        return cls(Camp.A, movement_lst=a_cmd_lst), cls(Camp.B, movement_lst=b_cmd_lst)


def generate_movements(chess, camp):
    """All possible movements that will not put King in danger"""

    for pi in chess.pieces(camp):
        pi: Piece = pi
        for mv in pi.movement_lst(chess):
            if validate_movement(chess, mv):
                yield mv


class ChessResult(object):
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.winner == other.winner


class Stalemate(ChessResult):
    winner = None
    loser = None

    def format(self, camp=None):
        return 'Draw'

    def __str__(self):
        return self.format()


class Checkmate(ChessResult):
    def __init__(self, winner: Camp):
        self.winner = winner
        self.loser = winner.another

    def format(self, camp=None):
        if camp is None:
            return '{} win the game'.format(self.winner)
        elif camp == self.winner:
            return 'You win'
        elif camp == self.loser:
            return 'You lose'
        else:
            raise ValueError

    def __str__(self):
        return self.format()


class ResignRequest(object):
    pass


class Resign(ChessResult):
    def __init__(self, resigner):
        self.winner = resigner.another
        self.loser = resigner

    def format(self, camp=None):
        if camp is None:
            return '{} resigned'.format(self.loser)
        elif camp == self.winner:
            return 'You win the game as your opponent resigned'
        elif camp == self.loser:
            return 'You resigned and lose the game'
        else:
            raise ValueError

    def __str__(self):
        return self.format()


def piece_lst_attacks(chess, sq: Square, camp=None) -> Iterator[Piece]:
    """Find out all pieces from <camp> that attacks <sq>"""
    for pi in chess.pieces(camp=camp):
        for att_sq in pi.attack_lst(chess):
            if att_sq == sq:
                yield pi
                break


def play(player_a, player_b):
    chess = Chess.setup()

    for player in itertools.cycle([player_a, player_b]):
        chess.turn = player.camp

        print()
        print(chess.format())

        # See if checkmate or stalemate happens
        #
        # + if the other camp have no movement to save their king -- checkmate
        # + if the other camp have no possible movement, and their king is not in danger -- stalemate
        #
        # Here we need:
        #    generate_movements(chess: Chess, camp: Camp) -> List[Movement]
        #    is_king_in_danger(chess: Chess, camp: Camp) -> bool

        if not list(generate_movements(chess, camp=chess.turn)):
            # no possible movements anymore

            ki_sq, ki_pi = chess.find_king()
            if any(piece_lst_attacks(chess, sq=ki_sq, camp=chess.turn.another)):
                return Checkmate(chess.turn.another)
            else:
                return Stalemate()

        mv = player(chess)
        if isinstance(mv, ResignRequest):
            return Resign(resigner=chess.turn)

        assert isinstance(mv, Movement)
        rule_check = validate_movement(chess, mv)

        while not rule_check.ok:
            print(rule_check)
            print()

            mv = player(chess)
            assert isinstance(mv, Movement)
            rule_check = validate_movement(chess, mv)

        chess = mv.apply_to(chess)


if __name__ == '__main__':
    print(play(Player(Camp.A), Player(Camp.B)))
