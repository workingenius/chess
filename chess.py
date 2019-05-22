import itertools
from enum import Enum
from string import ascii_lowercase
from typing import Iterator, Optional, List

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

    def format(self):
        return str(self)

    def attack_lst(self, chess) -> Iterator[Square]:
        """generate all squares it is able to capture in the next movement,
        even if there's not a piece or the piece is of same camp"""
        raise NotImplementedError

    def movement_lst(self, chess) -> Iterator['Movement']:
        """generate all movements launched by the piece"""
        raise NotImplementedError

    def is_valid_movement(self, chess, mv: 'Movement') -> bool:
        """if the movement <mv> about the job is valid. If not, tell reason"""
        raise NotImplementedError


class PKing(Piece):
    job = Job.KING

    def attack_lst(self, chess) -> Iterator[Square]:
        # squares surrounding it in all eight direction on board
        sq = chess.piece_to_square.get(self)
        return _on_board(sq + di for di in ALL_DIRECTIONS)

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
        for to in self.attack_lst(chess):
            if _has_enemy(chess, self.camp, at=to):
                yield Movement(frm=frm, capture=to)
            elif _has_alias(chess, self.camp, at=to):
                pass
            else:
                yield Movement(frm=frm, to=to)

        # TODO: CASTLING

    def is_valid_movement(self, chess, mv: 'Movement') -> 'RuleStatus':
        rule = 'King only move straight by one square'

        try:
            delta = mv.to - mv.frm
            passes(start=mv.frm, delta=delta, straight_only=True)
        except InvalidPath:
            return RuleBroken(rule)

        if delta.dis >= 2:
            return RuleBroken(rule)

        # TODO: check CASTLING

        return RULE_OK


class PPawn(Piece):
    job = Job.PAWN

    def attack_lst(self, chess) -> Iterator[Square]:
        # one square ahead of columns on both side
        # special case: en passant
        #   If a pawn is beside an enemy pawn who has just charged two square,
        #   he can move diagonally as usual and attack back to capture the enemy pawn

        return _on_board([
            chess.piece_to_square[self] + Delta.as_camp(forward=1, leftward=1),
            chess.piece_to_square[self] + Delta.as_camp(forward=1, rightward=1)
        ])

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

    def is_valid_movement(self, chess, mv: 'Movement') -> bool:
        pass


class PKnight(Piece):
    job = Job.KNIGHT


class PQueen(Piece):
    job = Job.QUEEN


class PCastle(Piece):
    job = Job.CASTLE


class PBishop(Piece):
    job = Job.BISHOP


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


class Delta(object):
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

    @classmethod
    def as_camp(cls, forward=None, backward=None, leftward=None, rightward=None, camp=Camp.A):
        assert (forward and not backward) or (not forward and backward), 'forward and backward cant both exist'
        assert (leftward and not rightward) or (not leftward and rightward), 'leftward and rightward cant both exist'

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
        return max(self.x, self.y)

    def __mul__(self, other):
        return self.__class__(x=self.x * other, y=self.y * other)

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


def _validate_movement(chess, mv: Movement):
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
    pass


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
    try:
        delta = mv.to - mv.frm
        pss = passes(start=mv.frm, delta=delta, straight_only=False)

    except InvalidPath:
        return RuleBroken('No piece can fly')

    if not delta.moved():
        return RuleBroken('The piece doesnt move')

    for pss_sq in pss:
        _pi = chess.square_to_piece.get(pss_sq)
        if _pi:
            return RuleBroken('Another piece is in the way')

    target_sq = mv.to
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
        di = delta
        if not (di.is_vertical or di.is_horizontal):
            return RuleBroken('Castle can only move horizontally or vertically')

        return goto_capture() or check_replace() or RuleOK()

    def check_queen():
        # no need to check it's direction
        # because queen can go in any direction as long as it does not fly
        return goto_capture() or check_replace() or RuleOK()

    def check_bishop():
        di = delta
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
        if not delta.is_forward(camp=chess.turn, just=False):
            return RuleBroken('A pawn can only proceed')

        if delta.dis > 2:
            return RuleBroken('The pawn moves too fast')

        if delta.is_forward(camp=chess.turn, just=True):
            # is proceeding

            # check starting charge
            if delta.dis == 2:
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
            if delta.dis == 2:
                return RuleBroken('A pawn can only attack enemy in diagonal direction by one square')

            # TODO: check en passant
            return goto_capture() or RuleOK()

        return RuleOK()

    def check_knight():
        if not delta.is_l_shape:
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

        # See if checkmate or stalemate happens
        #
        # + if the other camp have no movement to save their king -- checkmate
        # + if the other camp have no possible movement, and their king is not in danger -- stalemate
        #
        # Here we need:
        #    generate_movements(chess: Chess, camp: Camp) -> List[Movement]
        #    is_king_in_danger(chess: Chess, camp: Camp) -> bool


if __name__ == '__main__':
    play(Player(Camp.A), Player(Camp.B))
