import paddle
arg_1 = 0
arg_2 = -4
res = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
