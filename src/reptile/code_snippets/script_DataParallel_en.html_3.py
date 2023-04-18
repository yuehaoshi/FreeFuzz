# required: distributed
import paddle
import paddle.nn as nn
import paddle.distributed as dist

class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self._linear = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear(x)

dist.init_parallel_env()
model = SimpleNet()
dp_model = paddle.DataParallel(model)

inputs_1 = paddle.randn([10, 10], 'float32')
inputs_2 = paddle.ones([10, 10], 'float32')

with dp_model.no_sync():
    # gradients will not be synchronized
    dp_model(inputs_1).backward()

# synchronization happens here
dp_model(inputs_2).backward()