import paddle
import paddle.nn as nn
from paddle.static import InputSpec

device = paddle.set_device('cpu') # or 'gpu'

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.Tanh(),
    nn.Linear(200, 10))

input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(net, input, label)
optim = paddle.optimizer.SGD(learning_rate=1e-3,
    parameters=model.parameters())
model.prepare(optim,
            paddle.nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy())
data = paddle.rand((4, 784), dtype="float32")
label = paddle.randint(0, 10, (4, 1), dtype="int64")
loss, acc = model.eval_batch([data], [label])
print(loss, acc)
# [array([2.8825705], dtype=float32)] [0.0]