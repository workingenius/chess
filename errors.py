
class ChessError(Exception):
    pass


class BaseMovementError(ChessError):
    pass


class RuleBroken(ChessError):
    pass
