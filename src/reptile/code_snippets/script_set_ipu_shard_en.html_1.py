# required: ipu

import paddle
paddle.enable_static()
a = paddle.static.data(name='data', shape=[None, 1], dtype='float32')
relu = paddle.nn.ReLU()
relu = paddle.static.set_ipu_shard(relu, index=1, stage=1)
relu(a)