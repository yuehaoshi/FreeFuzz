# required: ipu

import paddle
import paddle.static as static

linear = paddle.nn.Linear(10, 10)
optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                 parameters=linear.parameters())
ipu_strategy = static.IpuStrategy()
ipu_strategy.set_optimizer(optimizer)