import itertools
import random
import re
import time
from enum import Enum
from functools import wraps
from string import ascii_lowercase
from typing import Iterator, List, Tuple

BOARD_SIZE = 8


def rule_validator(func):
    """Tag a function as a rule validator,

    which should return RuleOk object if rule is ok, and raise RuleBroken if it does not
    """
    return func


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

    def __add__(self, delta: 'Delta'):
        return self.__class__(x=(self.x + delta.x), y=(self.y + delta.y))

    def __sub__(self, sq: 'Square'):
        return Delta(x=(self.x - sq.x), y=(self.y - sq.y))

    def is_on_board(self):
        return 0 <= self.x < BOARD_SIZE and 0 <= self.y < BOARD_SIZE


def within_board(f):
    @wraps(f)
    def ff(*args, **kwargs):
        return filter(lambda x: x.is_on_board(), f(*args, **kwargs))

    return ff


class Piece(object):
    @property
    def job(self):
        raise NotImplementedError

    @property
    def abbr_char(self):
        raise NotImplementedError

    @property
    def abbr(self):
        return self.abbr_char.upper() if self.camp == Camp.A else self.abbr_char.lower()

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

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        """if the movement <mv> about the job is valid. If not, tell reason"""
        raise NotImplementedError

    def generate_movements(self, chess) -> Iterator['Movement']:
        """generate all movements launched by the piece"""
        raise NotImplementedError

    def attack_lst(self, chess) -> Iterator[Square]:
        """generate all squares it is able to capture in the next movement,
        even if there's not a piece or the piece is of same camp"""
        raise NotImplementedError


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
    def setup(cls, initial=None):

        def s(name):
            return Square.by_name(name)

        c = Chess()

        initial = initial or [
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

        for loc in initial:
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
        self.turn_count_without_capture = 0

    def copy(self):
        c = self.__class__()
        c.square_to_piece = dict(self.square_to_piece)
        c.piece_to_square = dict(self.piece_to_square)
        c.turn = self.turn
        c.history = list(self.history)
        return c

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

    def apply(self, mv, into_history=True):
        if self.square_to_piece.get(mv.frm) is None:
            raise mv.MovementError('{} is empty so nothing to move'.format(mv.frm.format()))

        if not mv.capture and mv.to and self.square_to_piece.get(mv.to):
            raise mv.MovementError('{} is occupied by another piece'.format(mv.to.format()))

        if mv.capture and self.square_to_piece.get(mv.capture) is None:
            raise mv.MovementError('{} is empty so nothing to capture'.format(mv.capture.format()))

        if mv.replace and mv.replace in self.piece_to_square:
            raise mv.MovementError('the piece to replace is on the board: {}'.format(mv.replace.format()))

        pie = self.remove(square=mv.frm)

        if mv.capture:
            self.remove(square=mv.capture)

        self.put(square=mv.to or mv.capture, piece=mv.replace or pie)

        if mv.sub_movement:
            self.apply(mv.sub_movement, into_history=False)

        if into_history:
            self.history.append(self)

        if not mv.capture:
            self.turn_count_without_capture += 1
        else:
            self.turn_count_without_capture = 0

        return self

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

        def piece_abbr(sq, p=None):
            c = '.' if sq.color == Color.WHITE else '|'
            if p is None:
                return c + c
            else:
                return p.abbr + c

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

    class MovementError(Exception):
        pass

    def __init__(self, frm, to=None, capture=None, replace=None, sub_movement: 'Movement' = None):
        self.frm = frm

        if to is None and capture is None:
            raise self.MovementError('either move to or capture should given')

        # two arguments for move from and move to respectively,
        # for the sake of "En passant"
        self.to = to or capture
        self.capture = capture

        # replace is designed for pawn promotion
        self.replace = replace

        self.sub_movement = sub_movement

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


class MLongCastling(Movement):
    def __init__(self, camp: Camp):
        if camp == Camp.A:
            frm = Square.by_name('e1')
            to = Square.by_name('c1')
            rook_frm = Square.by_name('a1')
            rook_to = Square.by_name('d1')

        elif camp == Camp.B:
            frm = Square.by_name('e8')
            to = Square.by_name('c8')
            rook_frm = Square.by_name('a8')
            rook_to = Square.by_name('d8')

        else:
            raise ValueError

        super(MLongCastling, self).__init__(
            frm=frm,
            to=to,
            sub_movement=Movement(
                frm=rook_frm,
                to=rook_to
            )
        )

    def format(self):
        return 'O-O-O'


class MShortCastling(Movement):
    def __init__(self, camp: Camp):
        if camp == Camp.A:
            frm = Square.by_name('e1')
            to = Square.by_name('g1')
            rook_frm = Square.by_name('h1')
            rook_to = Square.by_name('f1')

        elif camp == Camp.B:
            frm = Square.by_name('e8')
            to = Square.by_name('g8')
            rook_frm = Square.by_name('h8')
            rook_to = Square.by_name('f8')

        else:
            raise ValueError

        super(MShortCastling, self).__init__(
            frm=frm,
            to=to,
            sub_movement=Movement(
                frm=rook_frm,
                to=rook_to
            )
        )

    def format(self):
        return 'O-O'


def guess_movement(command_text, chess):
    cmd = re.sub(r'\s', '', command_text).lower()

    # remove comments
    if '#' in cmd:
        cmd = cmd.split('#')[0]

    if cmd == 'o-o':
        return MShortCastling(camp=chess.turn)

    elif cmd == 'o-o-o':
        return MLongCastling(camp=chess.turn)

    rep = None
    if '=' in cmd:
        move, rep = cmd.split('=')

        for j in [PKing, PQueen, PPawn, PCastle, PBishop, PKnight]:
            if rep == j.abbr_char.lower():
                rep = j.job
                break
        else:
            raise ValueError('={} does not represent any job'.format(rep))

        cmd = move

    segs = re.split(r'[-x*]', cmd)

    # from and to
    if len(segs) != 2:
        raise ValueError('Can not understand ' + cmd)
    frm, to = segs
    frm = Square.by_name(frm)
    to = Square.by_name(to)

    # find in all valid movements
    for _mv in generate_movements(chess, camp=chess.turn):
        if _mv.frm == frm and _mv.to == to:
            # if found one, then that's it
            return _mv

    # if not found, that's an invalid movement
    # but it's not our job to complain here, it's done in validation procedure
    if has_piece(chess, at=to):
        mv = Movement(frm=frm, capture=to, replace=rep)
    else:
        mv = Movement(frm=frm, to=to, replace=rep)

    return mv


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


# Implement path
#
# If two different squares <start> and <end> is in same file (column), or same rank (line), or same diagonal, there is a
# path between them. It means, a piece may go straight from <start> to <end>, without interfering piece into account.
#
# This concept is useful:
# 1. check moving is straight or not
# 2. check interfering piece


class InvalidPath(ValueError):
    pass


def make_path(start: Square, end: Square = None, delta: Delta = None, straight_only=True) -> List[Square]:
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


def has_interfering_piece(chess, path):
    """if some piece is in the way"""

    for sq in path:
        if chess.square_to_piece.get(sq):
            return True
    return False


# Implement chess rules
#
# Chess rules have two sides.
# 1. check if a movement is valid of not, and give reason which rule is broken
# 2. generate all possible movements
#
# Both these two sides are mainly delegated to subclass of Piece, like PKing, PQueen etc. , as rules of chess
# are strongly related to piece and job. Just like prototype of abstract class Piece shows, each Piece has several
# methods related to rule:
#
#     @rule_validator
#     def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
#         # side 1, validate a movement
#         pass
#
#     def generate_movements(self, chess) -> Iterator['Movement']:
#         # side 2, generate all movements related to the current piece
#         pass
#
#     def attack_lst(self, chess) -> Iterator[Square]:
#         # list all squares it can capture suppose there stand an enemy piece
#         pass
#
# All rule validators are decorated by "rule_validator", and they should raise RuleBroken exception with reason.


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


@rule_validator
def validate_movement(chess, mv: Movement):
    # 1. You have to move a piece, but not move air
    # 2. You can't move piece out of board
    # 3. The piece you move must be of your own camp, you can't move your partner's piece

    # 4. The movement must comply move rule of the job ... It differs from the job , but generally speaking,
    #   a. If you want to capture another piece, make sure there's one to capture
    #   b. If you capture a piece, it can't be of your own camp

    # 5. If your king is currently in danger, the movement must save your king.
    #     (note: If the king is currently in danger, and none of possible movements is able to save him, it's checkmate
    #     and should be found right after last movement.)
    # 6. And make sure that the movement will not expose your king to danger.

    pi: Piece = chess.square_to_piece.get(mv.frm)
    if pi is None:
        raise RuleBroken('There is not a piece')

    if not mv.to.is_on_board():
        raise RuleBroken('You cant move out of board')

    if pi.camp != chess.turn:
        raise RuleBroken('You can\'t move your partner\'s piece')

    # special logic for castling
    if isinstance(mv, (MLongCastling, MShortCastling)):
        _, king_pi = chess.find_king(camp=chess.turn)
        pi = king_pi

    pi.validate_movement(chess, mv)

    # check King safety

    chess = chess.copy()
    chess = chess.apply(mv)
    king_loc, _ = chess.find_king(camp=chess.turn)

    # traverse piece from other camp
    if is_under_attack(chess, sq=king_loc, by_camp=chess.turn.another):
        raise KingInDanger('King in danger')

    return RULE_OK


def generate_movements(chess, camp) -> Iterator[Movement]:
    """All possible movements that will not put King in danger"""

    for pi in chess.pieces(camp):
        pi: Piece = pi
        for mv in pi.generate_movements(chess):
            try:
                validate_movement(chess, mv)
            except RuleBroken:
                pass
            else:
                yield mv


def is_under_attack(chess, sq: Square, by_camp=None) -> bool:
    """check if square <sq> is under attack by camp <by_camp>"""
    for pi in chess.pieces(camp=by_camp):
        for att_sq in pi.attack_lst(chess):
            if att_sq == sq:
                return True
    return False


def has_enemy(chess, camp, at: Square):
    return chess.square_to_piece.get(at) and chess.square_to_piece.get(at).camp != camp


def has_alias(chess, camp, at: Square):
    return chess.square_to_piece.get(at) and chess.square_to_piece.get(at).camp == camp


def has_piece(chess, at: Square):
    return bool(chess.square_to_piece.get(at))


@rule_validator
def is_valid_capture(chess, capture, camp):
    ca = chess.square_to_piece.get(capture)

    if ca is None:
        raise RuleBroken('There is no piece to capture')

    if ca.camp != camp:
        pass
    else:
        raise RuleBroken('You can not capture piece of your own camp')


@rule_validator
def is_valid_move_to(chess, move: Square):
    if has_piece(chess, at=move):
        raise RuleBroken('You can not move to a square with piece')


@rule_validator
def is_clear_path(chess, path: Iterator[Square]):
    if has_interfering_piece(chess, path):
        raise RuleBroken('You can not move over other pieces')


def move_to_or_capture(chess, frm, to, camp):
    if has_enemy(chess, camp=camp, at=to):
        return Movement(frm=frm, capture=to)
    elif has_alias(chess, camp=camp, at=to):
        return
    else:
        return Movement(frm=frm, to=to)


def make_movements_by_target(chess, frm, to_lst, camp):
    for to in to_lst:
        mv = move_to_or_capture(chess, frm, to, camp)
        if mv:
            yield mv


def move_down_straight(chess, square, direction):
    """generate squares along a direction <direction> from a start point <square>,
    and stops till a interfering piece or chess border is met"""

    hidden = False

    for dis in range(1, BOARD_SIZE):
        if hidden:
            break

        delta = direction * dis
        to = square + delta

        if has_piece(chess, at=to):
            hidden = True
            yield to

        else:
            yield to


class PKing(Piece):
    abbr_char = 'K'
    job = Job.KING

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        # squares surrounding it in all eight direction on board
        sq = chess.piece_to_square.get(self)
        for di in ALL_DIRECTIONS:
            yield sq + di

    def generate_movements(self, chess) -> Iterator['Movement']:
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
        for mv in make_movements_by_target(chess, frm, self.attack_lst(chess), camp=self.camp):
            yield mv

        cas = MLongCastling(camp=self.camp)
        try:
            self.validate_castling(chess, cas)
        except RuleBroken:
            pass
        else:
            yield cas

        cas = MShortCastling(camp=self.camp)
        try:
            self.validate_castling(chess, cas)
        except RuleBroken:
            pass
        else:
            yield cas

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        if isinstance(mv, (MLongCastling, MShortCastling)):
            return self.validate_castling(chess, mv)

        else:
            king_rule_caption = 'King only move straight by one square'

            try:
                delta = mv.to - mv.frm
                make_path(start=mv.frm, delta=delta, straight_only=True)
            except InvalidPath:
                raise RuleBroken(king_rule_caption)

            if delta.dis >= 2:
                raise RuleBroken(king_rule_caption)

            if mv.capture:
                is_valid_capture(chess, mv.capture, camp=self.camp)
            else:
                is_valid_move_to(chess, mv.to)

        return RULE_OK

    @rule_validator
    def validate_castling(self, chess, cas: 'Movement'):
        king_loc = cas.frm
        rook_loc = cas.sub_movement.frm

        if has_piece(chess, king_loc):
            if chess.square_to_piece[king_loc].job != Job.KING:
                raise RuleBroken('King has moved')
        else:
            raise RuleBroken('King has moved')

        if has_piece(chess, rook_loc):
            if chess.square_to_piece[rook_loc].job != Job.CASTLE:
                raise RuleBroken('Castle has moved')
        else:
            raise RuleBroken('Castle has moved')

        path = make_path(start=king_loc, end=rook_loc)

        # if interfered by other pieces
        try:
            is_clear_path(chess, path)
        except RuleBroken:
            raise RuleBroken('Castling interfered by other pieces')

        # king or rook in currently attacked
        if is_under_attack(chess, sq=king_loc, by_camp=self.camp.another) or \
                is_under_attack(chess, sq=rook_loc, by_camp=self.camp.another):
            raise RuleBroken('You cannot do castling when king or rook is attacked')

        # danger in path way
        for psq in path:
            if is_under_attack(chess, sq=psq, by_camp=self.camp.another):
                raise RuleBroken('Castling requires a totally safe path')

        # TODO: ensure both rook and king is not moved

        return RULE_OK


class PPawn(Piece):
    abbr_char = 'P'
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

    def generate_movements(self, chess) -> Iterator['Movement']:
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
            d = Delta.as_camp(forward=2, camp=self.camp)
            to = at + d
            if not has_interfering_piece(chess, make_path(at, delta=d)) and not has_piece(chess, at=to):
                # no need to check for promotion in starting point
                yield Movement(frm=at, to=to)

        # one step forward
        d = Delta.as_camp(forward=1, camp=self.camp)
        to = at + d
        if not has_piece(chess, at=to):
            for mv in mv_with_promotion(frm=at, to=to):
                yield mv

        # capture
        for cp in self.attack_lst(chess):
            if has_enemy(chess, camp=self.camp, at=cp):
                for mv in mv_with_promotion(frm=at, capture=cp):
                    yield mv

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':

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
                path = make_path(start=mv.frm, delta=delta)
                is_clear_path(chess, path)

            if mv.capture:
                raise RuleBroken('Pawns cant attack ahead')
            else:
                is_valid_move_to(chess, mv.to)

        else:
            # is attacking
            pawn_cap_rule_caption = 'A pawn can only attack enemy in diagonal direction by one square'

            if mv.capture is None:
                raise RuleBroken(pawn_cap_rule_caption)

            if delta.dis != 1:
                raise RuleBroken(pawn_cap_rule_caption)

            is_valid_capture(chess, capture=mv.capture, camp=self.camp)

            # TODO: check en passant

        return RULE_OK


class PKnight(Piece):
    abbr_char = 'N'
    job = Job.KNIGHT

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for d in L_SHAPES:
            yield chess.piece_to_square[self] + d

    def generate_movements(self, chess) -> Iterator['Movement']:
        return make_movements_by_target(chess, chess.piece_to_square[self], self.attack_lst(chess), camp=self.camp)

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        delta = mv.to - mv.frm
        if not delta.is_l_shape:
            raise RuleBroken('Knight can only move in l shape')

        if mv.capture:
            is_valid_capture(chess, mv.capture, camp=self.camp)
        else:
            is_valid_move_to(chess, mv.to)

        return RULE_OK


class PQueen(Piece):
    abbr_char = 'Q'
    job = Job.QUEEN

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in ALL_DIRECTIONS:
            for to in move_down_straight(chess, chess.piece_to_square[self], dir):
                yield to

    def generate_movements(self, chess) -> Iterator['Movement']:
        return make_movements_by_target(chess, chess.piece_to_square[self], self.attack_lst(chess), camp=self.camp)

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        # valid direction
        try:
            pss = make_path(start=mv.frm, end=mv.to, straight_only=True)
        except InvalidPath:
            raise RuleBroken('Queen can only move straightly')

        # valid moving path
        is_clear_path(chess, pss)

        if mv.capture:
            is_valid_capture(chess, mv.capture, camp=self.camp)
        else:
            is_valid_move_to(chess, mv.to)

        return RULE_OK


class PCastle(Piece):
    abbr_char = 'R'
    job = Job.CASTLE

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in HORIZONTAL_AND_VERTICAL_DIRECTIONS:
            for to in move_down_straight(chess, chess.piece_to_square[self], dir):
                yield to

    def generate_movements(self, chess) -> Iterator['Movement']:
        return make_movements_by_target(chess, chess.piece_to_square[self], self.attack_lst(chess), camp=self.camp)

    @rule_validator
    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        castle_rule_caption = 'Castle can only move in vertical or horizontal lines'

        # valid direction
        try:
            delta = mv.to - mv.frm
            pss = make_path(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            raise RuleBroken(castle_rule_caption)

        if delta.is_horizontal or delta.is_vertical:
            pass
        else:
            raise RuleBroken(castle_rule_caption)

        # valid moving path
        is_clear_path(chess, pss)

        if mv.capture:
            is_valid_capture(chess, mv.capture, camp=self.camp)
        else:
            is_valid_move_to(chess, mv.to)

        return RULE_OK


class PBishop(Piece):
    abbr_char = 'B'
    job = Job.BISHOP

    @within_board
    def attack_lst(self, chess) -> Iterator[Square]:
        for dir in DIAGONAL_DIRECTIONS:
            for to in move_down_straight(chess, chess.piece_to_square[self], dir):
                yield to

    def generate_movements(self, chess) -> Iterator['Movement']:
        return make_movements_by_target(chess, chess.piece_to_square[self], self.attack_lst(chess), camp=self.camp)

    def validate_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        castle_rule_caption = 'Bishop can only move in diagonal lines'

        # valid direction
        try:
            delta = mv.to - mv.frm
            pss = make_path(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            raise RuleBroken(castle_rule_caption)

        if delta.is_diagonal:
            pass
        else:
            raise RuleBroken(castle_rule_caption)

        # valid moving path
        is_clear_path(chess, pss)

        if mv.capture:
            is_valid_capture(chess, mv.capture, camp=self.camp)
        else:
            is_valid_move_to(chess, mv.to)

        return RULE_OK


# Implement Players
# Players come up with movements according to chess situation.


class Player(object):
    def __init__(self, camp):
        self._camp = camp

    def __call__(self, chess):
        try:
            cmd = input('{} turn >> '.format(self.camp)).lower()
        except EOFError:
            cmd = 'resign'

        return self.parse_command(cmd, chess)

    @staticmethod
    def parse_command(cmd, chess):
        cmd = cmd.lower().strip()

        if 'resign' in cmd:
            return ResignRequest()
        else:
            return guess_movement(cmd, chess=chess)

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
            return self.parse_command(self.movement_lst[self.cur - 1], chess)
        else:
            return ResignRequest()

    @classmethod
    def pair_by_manual_text(cls, text):
        cmd_lst = [t.strip() for t in text.split('\n') if t.strip()]
        a_cmd_lst = [cmd for i, cmd in enumerate(cmd_lst) if i % 2 == 0]
        b_cmd_lst = [cmd for i, cmd in enumerate(cmd_lst) if i % 2 == 1]
        return cls(Camp.A, movement_lst=a_cmd_lst), cls(Camp.B, movement_lst=b_cmd_lst)


class RandomPlayer(Player):
    def __call__(self, chess):
        mv_lst = list(generate_movements(chess, self.camp))
        return mv_lst[random.randint(0, len(mv_lst) - 1)]


# Implement game procedure


class ChessResult(object):
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.winner == other.winner


class Draw(ChessResult):
    winner = None
    loser = None

    def format(self, camp=None):
        return 'Draw'

    def __str__(self):
        return self.format()


class Stalemate(Draw):
    def format(self, camp=None):
        return 'Draw by stalemate'


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


def play(player_a, player_b) -> ChessResult:
    chess = Chess.setup()

    for player in itertools.cycle([player_a, player_b]):
        chess.turn = player.camp

        print()
        print(chess.format())

        # After 50 steps without capture, game draw
        if chess.turn_count_without_capture >= 50:
            return Draw()

        # When one camp has no valid movement, either checkmate or stalemate happens
        #
        # + if the other camp have no movement to save their king -- checkmate
        # + if the other camp have no possible movement, and their king is not in danger -- stalemate

        if not list(generate_movements(chess, camp=chess.turn)):
            # no possible movements anymore

            # check if king is in danger
            ki_sq, ki_pi = chess.find_king()
            if is_under_attack(chess, sq=ki_sq, by_camp=chess.turn.another):
                return Checkmate(chess.turn.another)
            else:
                return Stalemate()

        mv = player(chess)
        if isinstance(mv, ResignRequest):
            return Resign(resigner=chess.turn)
        assert isinstance(mv, Movement)

        while True:
            try:
                validate_movement(chess, mv)
            except RuleBroken as rb:
                print(rb)
                print()

                mv = player(chess)
                if isinstance(mv, ResignRequest):
                    return Resign(resigner=chess.turn)
                assert isinstance(mv, Movement)
            else:
                break

        chess = chess.apply(mv)


if __name__ == '__main__':
    print(play(Player(Camp.A), RandomPlayer(Camp.B)))
