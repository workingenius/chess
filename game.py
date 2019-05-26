from datetime import datetime

from chess import Camp, RandomPlayer, Chess, Movement, validate_movement, RuleBroken, check_outcome, ChessResult

message_box = []


def keep_deliver():
    while message_box:
        assert len(message_box) < 100
        msg, targ = message_box.pop(0)
        targ.process_message(msg)


def send(message, to):
    message_box.append((message, to))


class Message(object):
    pass


class MsgStart(Message):
    pass


class MsgSituation(Message):
    def __init__(self, chess):
        self.chess = chess


class MsgYourTurn(Message):
    def __init__(self, chess):
        self.chess = chess


class MsgMovement(Message):
    def __init__(self, mv, camp):
        self.mv: Movement = mv
        self.camp = camp

    def __str__(self):
        return '{camp} {mv}'.format(
            camp=self.camp,
            mv=self.mv
        )


class MsgInvalidMovement(Message):
    def __init__(self, chess, rb):
        self.chess = chess
        self.rb = rb


class MsgOutcome(Message):
    def __init__(self, chess, outcome):
        self.chess = chess
        self.outcome: ChessResult = outcome


class Game(object):
    def __init__(self, client_a, client_b):
        self.client_a = client_a
        self.client_b = client_b
        self.audience = []

        self.chess = None
        self.turn = None
        self.cur_client = None

    def process_message(self, msg):
        if isinstance(msg, MsgStart):
            self.chess = Chess.setup()
            self.chess.turn = Camp.A
            self.turn = Camp.A
            self.cur_client = self.client_a

            self.publish(MsgSituation(chess=self.chess))
            send(MsgYourTurn(self.chess), self.cur_client)

        elif isinstance(msg, MsgMovement):
            assert self.turn == msg.camp

            mv = msg.mv
            try:
                validate_movement(self.chess, mv)
            except RuleBroken as e:
                send(MsgInvalidMovement(e, self.chess), self.cur_client)

            else:
                self.chess = self.chess.apply(mv)
                self.switch()

                self.publish(msg)
                self.publish(MsgSituation(self.chess))

                ot = check_outcome(self.chess)
                if ot:
                    self.publish(MsgOutcome(self.chess, ot))

                else:
                    send(MsgYourTurn(chess=self.chess), self.cur_client)

    def publish(self, msg):
        send(msg, self.client_a)
        send(msg, self.client_b)
        for au in self.audience:
            send(msg, au)

    def switch(self):
        self.chess.turn = self.chess.turn.another
        self.turn = self.turn.another
        self.cur_client = self.client_a if self.cur_client is self.client_b else self.client_b


class Client(object):
    def __init__(self, camp, player):
        self.camp = camp
        self.player = player
        self.game = None

    def process_message(self, msg):
        if isinstance(msg, MsgYourTurn):
            mv = self.player(msg.chess)
            send(MsgMovement(mv, camp=self.camp), self.game)

        if isinstance(msg, MsgInvalidMovement):
            mv = self.player(msg.chess)
            send(MsgMovement(mv, camp=self.camp), self.game)


class Printer(object):
    def process_message(self, msg):
        if isinstance(msg, MsgMovement):
            print('{time} {mv}'.format(time=datetime.now(), mv=msg))
        elif isinstance(msg, MsgOutcome):
            print('{time} {ot}'.format(time=datetime.now(), ot=msg.outcome))
            print(msg.chess.format())


def play(client_a, client_b):
    game = Game(client_a, client_b)
    game.audience = [Printer()]
    client_a.game = game
    client_b.game = game
    send(MsgStart(), to=game)
    keep_deliver()


if __name__ == '__main__':
    play(
        client_a=Client(
            camp=Camp.A,
            player=RandomPlayer(Camp.A)
        ),
        client_b=Client(
            camp=Camp.B,
            player=RandomPlayer(Camp.B)
        )
    )
