# required: ipu

import paddle
paddle.enable_static()
a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
with paddle.static.ipu_shard_guard(index=0, stage=0):
    b = a + 1
with paddle.static.ipu_shard_guard(index=1, stage=1):
    c = b + 1
with paddle.static.ipu_shard_guard(index=0, stage=2):
    d = c + 1