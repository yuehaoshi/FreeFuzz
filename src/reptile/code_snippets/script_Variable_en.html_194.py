# scale as a float32 number
import paddle

data = paddle.randn(shape=[2,3], dtype='float32')
res = paddle.scale(data, scale=2.0, bias=1.0)