# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import chess_pb2 as chess__pb2


class ChessStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.subscribe = channel.unary_stream(
        '/chessmsg.Chess/subscribe',
        request_serializer=chess__pb2.SubscribeRequest.SerializeToString,
        response_deserializer=chess__pb2.Notification.FromString,
        )
    self.make_move = channel.unary_unary(
        '/chessmsg.Chess/make_move',
        request_serializer=chess__pb2.Action.SerializeToString,
        response_deserializer=chess__pb2.ActionStatus.FromString,
        )
    self.attend_game = channel.unary_unary(
        '/chessmsg.Chess/attend_game',
        request_serializer=chess__pb2.GameRequest.SerializeToString,
        response_deserializer=chess__pb2.GameResponse.FromString,
        )


class ChessServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def subscribe(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def make_move(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def attend_game(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ChessServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'subscribe': grpc.unary_stream_rpc_method_handler(
          servicer.subscribe,
          request_deserializer=chess__pb2.SubscribeRequest.FromString,
          response_serializer=chess__pb2.Notification.SerializeToString,
      ),
      'make_move': grpc.unary_unary_rpc_method_handler(
          servicer.make_move,
          request_deserializer=chess__pb2.Action.FromString,
          response_serializer=chess__pb2.ActionStatus.SerializeToString,
      ),
      'attend_game': grpc.unary_unary_rpc_method_handler(
          servicer.attend_game,
          request_deserializer=chess__pb2.GameRequest.FromString,
          response_serializer=chess__pb2.GameResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'chessmsg.Chess', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
