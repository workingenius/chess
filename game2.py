from collections import defaultdict
from threading import Semaphore, RLock
from uuid import uuid4

import chess as chess_mod
from chess import Chess, Movement, Camp
from proto.chess_pb2 import GameResponse, Movement as MovementMsg, ActionStatus, Notification, ChessSituation, \
    Camp as CampMsg, Action as ActionMsg, SPC
from proto.chess_pb2_grpc import ChessServicer as ChessServicerStub


def thread_safe(proc):
    lock = RLock()

    def nproc(*args, **kwargs):
        with lock:
            return proc(*args, **kwargs)

    return nproc


class Client(object):
    client_dct = {}

    def __init__(self, client_id):
        self.client_id = client_id
        self.client_dct[client_id] = self

    @classmethod
    def get_or_create(cls, client_id):
        if client_id in cls.client_dct:
            return cls.client_dct[client_id]
        else:
            return cls(client_id)

    def __eq__(self, other):
        return type(self) == type(other) and self.client_id == other.client_id

    def __hash__(self):
        return hash(self.client_id)


class ClientMatcher(object):
    def __init__(self):
        self.waiting = []
        self.matched = {}

        self.lock = RLock()

    def register_client_and_wait_for_match(self, cli):
        with self.lock:
            if self.waiting:
                sem, cli0 = self.waiting.pop(0)
                has_waiting = True
            else:
                sem = Semaphore(value=0)
                self.waiting.append((sem, cli))
                has_waiting = False

        if has_waiting:
            sem: Semaphore
            self.matched[cli0] = cli
            sem.release()
            return cli0, cli

        else:
            sem.acquire()
            return cli, self.matched.pop(cli)


matcher = ClientMatcher()


class Game(object):
    client_game_dct = {}
    id_game_dct = {}

    @classmethod
    @thread_safe
    def get_or_create(cls, client1, client2) -> 'Game':
        cli_key = cls.mk_cli_key(client1, client2)

        if cli_key in cls.client_game_dct:
            return cls.client_game_dct[cli_key]

        else:
            return Game(client1=client1, client2=client2)

    @classmethod
    def by_id(cls, game_id) -> 'Game':
        return cls.id_game_dct[game_id]

    @staticmethod
    def mk_cli_key(cli1, cli2):
        return frozenset([cli1, cli2])

    def __init__(self, client1, client2):
        self.client1: Client = client1
        self.client2: Client = client2
        self.id = uuid4()

        cli_key = self.mk_cli_key(client1, client2)
        self.client_game_dct[cli_key] = self

        self.id_game_dct[self.id] = self

        self.ready_client_dct = {
            client1: False,
            client2: False
        }

        self.lock = RLock()
        self.start_signal = Semaphore(value=0)

        self.chess = None
        self._is_playing = False

    @property
    def is_playing(self):
        with self.lock:
            return self._is_playing

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def wait_till_both_ready_and_start(self, client) -> None:

        with self.lock:
            self.ready_client_dct[client] = True
            ready = all(self.ready_client_dct.values())

        if ready:
            self.start_game()
            not_man.publish(self,
                            chess_to_message(self.chess))
            self.start_signal.release()

        else:
            self.start_signal.acquire()

    def start_game(self) -> None:
        with self.lock:
            self.chess = Chess.setup()
            self._is_playing = True

    def situation(self) -> Notification:
        return Notification(situation=chess_to_message(self.chess))

    def make_move(self, action: ActionMsg) -> ActionStatus:
        camp = camp_from_message(action.camp)
        assert self.chess.turn == camp

        if action.special == 'resign':
            self.chess.apply()
            chess_mod.ResignRequest()


def chess_to_message(chess: Chess) -> ChessSituation:
    spc_lst = []
    for sq, pi in chess.locations():
        spc = spc_to_message(sq=sq, pi=pi)
        spc_lst.append(spc)

    his_lst = []
    for mv in chess.history:
        his_lst.append(
            movement_to_message(mv)
        )

    return ChessSituation(
        turn=camp_to_message(chess.turn),
        spc=spc_lst,
        history=his_lst
    )


def piece_to_message(pi: chess_mod.Piece) -> int:
    i = None

    if pi.job == chess_mod.Job.PAWN:
        i = 1
    elif pi.job == chess_mod.Job.CASTLE:
        i = 2
    elif pi.job == chess_mod.Job.KNIGHT:
        i = 3
    elif pi.job == chess_mod.Job.BISHOP:
        i = 4
    elif pi.job == chess_mod.Job.QUEEN:
        i = 5
    elif pi.job == chess_mod.Job.KING:
        i = 6

    if pi.camp == Camp.A:
        i = i
    elif pi.camp == Camp.B:
        i = -i

    return i


def piece_from_message(pi_msg: int) -> chess_mod.Piece:
    camp = Camp.A if pi_msg > 0 else Camp.B

    j = None

    if pi_msg == 1:
        j = chess_mod.Job.PAWN
    elif pi_msg == 2:
        j = chess_mod.Job.CASTLE
    elif pi_msg == 3:
        j = chess_mod.Job.KNIGHT
    elif pi_msg == 4:
        j = chess_mod.Job.BISHOP
    elif pi_msg == 5:
        j = chess_mod.Job.QUEEN
    elif pi_msg == 6:
        j = chess_mod.Job.KING

    return chess_mod.cons_piece(camp=camp, job=j)


def square_to_message(sq: chess_mod.Square) -> int:
    return sq.x * 8 + sq.y


def square_from_message(sq_msg: int) -> chess_mod.Square:
    return chess_mod.Square(x=sq_msg // 8, y=sq_msg % 8)


def spc_to_message(sq, pi) -> SPC:
    p = piece_to_message(pi)
    s = square_to_message(sq)
    c = camp_to_message(sq.camp)
    return SPC(square=s, piece=p, camp=c)


def movement_to_message(mv: Movement) -> MovementMsg:
    pi = piece_to_message(mv.piece) if mv.piece else None
    frm = square_to_message(mv.frm) if mv.frm else None
    to = square_to_message(mv.to) if mv.to else None
    cap = square_to_message(mv.capture) if mv.capture else None
    rep = piece_to_message(mv.replace) if mv.replace else None
    sub = movement_to_message(mv.sub_movement) if mv.sub_movement else None
    return MovementMsg(piece=pi, frm=frm, to=to, capture=cap, replace=rep, sub=sub)


def movement_from_message(mv_msg: MovementMsg) -> Movement:
    return Movement(
        piece=piece_from_message(mv_msg.piece),
        frm=square_from_message(mv_msg.frm),
        to=square_from_message(mv_msg.to),
        capture=piece_from_message(mv_msg.capture),
        replace=piece_from_message(mv_msg.replace),
        sub_movement=movement_from_message(mv_msg.sub)
    )


def camp_from_message(cp_msg: CampMsg) -> Camp:
    if cp_msg == CampMsg.A:
        return Camp.A
    elif cp_msg == CampMsg.B:
        return Camp.B


def camp_to_message(camp) -> CampMsg:
    if camp == Camp.A:
        return CampMsg.A
    elif camp == Camp.B:
        return CampMsg.B


class NotificationManager(object):
    ENDING = StopIteration()

    def __init__(self):
        self.relation = defaultdict(list)
        self.message_lst = []

        self.lock = RLock()
        self.work_signal = Semaphore(value=0)

    def subscribe(self, game):
        sub = Subscriber(sem=Semaphore(value=0), game=game, nm=self)
        with self.lock:
            self.relation[game].append(sub)
        return sub

    def cancel(self, game, subscriber):
        with self.lock:
            self.relation[game].remove(subscriber)

    def publish(self, game, message):
        with self.lock:
            self.message_lst.append((game, message))
            self.work_signal.release()

    def keep_deliver(self):
        while True:
            self.work_signal.acquire()

            game, msg = self.message_lst.pop(0)
            if msg == self.ENDING:
                for sub in self.relation[game]:
                    sub.finished = True
                    sub.sem.release()
            else:
                for sub in self.relation[game]:
                    sub.msg_lst.append(msg)
                    sub.sem.release()


notification_manager = NotificationManager()
not_man = notification_manager


class Subscriber(object):
    def __init__(self, sem, game, nm: NotificationManager):
        self.sem: Semaphore = sem
        self.game = game
        self.nm = nm
        self.msg_lst = []
        self.finished = False

    def __iter__(self):
        while True:
            self.sem.acquire()
            msg = self.msg_lst.pop(0)
            yield msg

            if self.finished and not self.msg_lst:
                break

    def __del__(self):
        self.nm.cancel(self.game, self)


class ChessServicer(ChessServicerStub):
    def subscribe(self, request, context):
        game_id = request.game_id
        cli = Client.get_or_create(request.client_id)

        game = Game.by_id(game_id)

        is_player = (cli == game.client1 or cli == game.client2)
        is_audience = not is_player

        msg_stream = not_man.subscribe(game)

        if is_player:
            game.wait_till_both_ready_and_start(client=cli)

        elif is_audience:
            if game.is_playing:
                yield game.situation()

        for msg in msg_stream:
            yield msg

    def make_move(self, request, context):
        game = Game.by_id(request.game_id)
        return game.make_move(request.movement, special=request.special)

    def attend_game(self, request, context):
        cli = Client.get_or_create(request.client_id)
        cli0, cli1 = matcher.register_client_and_wait_for_match(cli)
        game = Game.get_or_create(client1=cli0, client2=cli1)
        return GameResponse(game_id=game.id)
