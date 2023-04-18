import paddle
import numpy as np

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # LayerList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x